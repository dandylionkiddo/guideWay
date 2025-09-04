"""
이 파일은 세그멘테이션 모델의 평가를 수행하는 `Evaluator` 클래스와
관련 유틸리티를 정의합니다.
"""
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.models.utils import resize


class SegIOU:
    """
    세그멘테이션을 위한 IoU(Intersection over Union) 계산 클래스.
    `torch.histc`를 사용하여 효율적으로 교집합(intersection)과 합집합(union)을 계산합니다.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        """
        Args:
            num_classes (int): 전체 클래스의 개수.
            ignore_index (int, optional): 계산에서 무시할 레이블 인덱스. Defaults to -1.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        모델 예측과 실제 레이블을 바탕으로 교집합과 합집합을 계산합니다.

        Args:
            outputs (torch.Tensor): 모델의 예측 결과 텐서. (B, H, W)
            targets (torch.Tensor): 실제 레이블 텐서. (B, H, W)

        Returns:
            dict[str, torch.Tensor]: 클래스별 교집합("i")과 합집합("u") 텐서를 담은 딕셔너리.
        """
        mask = (targets >= 0) & (targets < self.num_classes)

        # torch.histc는 float 텐서만 지원하므로, int 타입의 레이블을 float으로 변환합니다.
        # 또한 min=1로 설정하므로, 0번 클래스를 포함하기 위해 모든 값에 1을 더합니다.
        outputs_float = (outputs + 1).to(torch.float32)
        targets_float = (targets + 1).to(torch.float32)

        # 예측값과 실제값 각각에 대해 클래스별 픽셀 수를 계산합니다.
        outputs_hist = torch.histc(outputs_float * mask, bins=self.num_classes, min=1, max=self.num_classes)
        targets_hist = torch.histc(targets_float * mask, bins=self.num_classes, min=1, max=self.num_classes)

        # 교집합(intersection) 계산: 예측과 실제가 일치하는 픽셀만 계산합니다.
        intersection_tensor = (outputs == targets) * mask
        intersection_values = targets_float * intersection_tensor
        intersections_hist = torch.histc(intersection_values, bins=self.num_classes, min=1, max=self.num_classes)

        # 합집합(union) 계산: Union(A, B) = A + B - Intersection(A, B)
        unions = outputs_hist + targets_hist - intersections_hist

        return {"i": intersections_hist, "u": unions}


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple | list,
    opacity: float = 0.5,
) -> np.ndarray:
    """
    원본 이미지에 세그멘테이션 마스크를 시각적으로 오버레이한 이미지를 생성합니다.

    Args:
        image (np.ndarray): 원본 이미지 (H, W, C).
        mask (np.ndarray): 모델이 예측한 레이블 마스크 (H, W).
        colors (tuple | list): 클래스별 색상 팔레트.
        opacity (float, optional): 마스크의 투명도. Defaults to 0.5.

    Returns:
        np.ndarray: 마스크가 오버레이된 이미지.
    """
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # class_colors가 없는 경우(None)를 대비하여 기본 팔레트 생성
    if colors is None:
        # 랜덤 색상 팔레트 생성
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        if k < len(colors):
            seg_mask[mask == k, :] = color
            
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


class Evaluator:
    """
    세그멘테이션 모델의 평가를 총괄하는 클래스.
    추론, 성능 지표 계산, 결과 시각화 및 저장을 수행합니다.
    """

    def __init__(self, model: torch.nn.Module, data_loader: DataLoader, config: dict) -> None:
        """
        Args:
            model (torch.nn.Module): 평가할 모델.
            data_loader (DataLoader): 평가용 데이터로더.
            config (dict): 평가 설정 딕셔너리.
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.tasks = config["evaluation_tasks"]

        # 평가지표 계산을 위한 객체들 초기화
        ignore_index = getattr(self.data_loader.dataset, "ignore_index", -1)
        self.iou_metric = SegIOU(
            num_classes=len(self.data_loader.dataset.classes),
            ignore_index=ignore_index
        )
        self.intersection_meter = AverageMeter(is_distributed=False)
        self.union_meter = AverageMeter(is_distributed=False)
        self.time_meter = AverageMeter(is_distributed=False)

        self.run_dir = None
        self.labeled_dir = None

    def _setup_save_dirs(self) -> None:
        """결과를 저장할 디렉토리를 생성합니다."""
        base_save_path = self.config.get("save_path")
        if base_save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(base_save_path, timestamp)
            self.labeled_dir = os.path.join(self.run_dir, "labeled")
            os.makedirs(self.labeled_dir, exist_ok=True)
            print(f"Saving results to {self.run_dir}")

    def evaluate(self) -> dict[str, Any]:
        """
        전체 평가 파이프라인을 실행합니다.
        1. 데이터셋 전체에 대해 추론을 수행하며 성능 지표를 누적합니다.
        2. (선택) 추론 결과를 이미지 파일로 저장합니다.
        3. 최종 성능 지표를 요약하여 반환합니다.

        Returns:
            dict[str, Any]: mIoU, FPS 등 최종 평가 결과를 담은 딕셔너리.
        """
        self._setup_save_dirs()

        results_to_save = []
        printed_mask_shape = False
        with torch.inference_mode():
            with tqdm(total=len(self.data_loader), desc="Stage 1: Inference") as t:
                for feed_dict in self.data_loader:
                    images, mask = feed_dict["image"].cuda(), feed_dict["label"].cuda()

                    start_time = time.time()
                    output = self.model(images)
                    self.time_meter.update(time.time() - start_time)

                    if output.shape[-2:] != mask.shape[-2:]:
                        output = resize(output, size=mask.shape[-2:], mode="bilinear", align_corners=True)

                    pred = torch.argmax(output, dim=1)

                    if "calculate_miou" in self.tasks:
                        if not printed_mask_shape:
                            print(f"[Evaluator] Mask shape: {mask.shape}")
                            printed_mask_shape = True
                        stats = self.iou_metric(pred, mask)
                        self.intersection_meter.update(stats["i"])
                        self.union_meter.update(stats["u"])
                        iou_per_class = self.intersection_meter.sum / self.union_meter.sum
                        valid_classes_mask = self.union_meter.sum > 0
                        miou = iou_per_class[valid_classes_mask].cpu().mean().item() * 100 if valid_classes_mask.sum() > 0 else 0.0
                        t.set_postfix({"mIOU": f"{miou:.2f}%"})

                    t.update()

                    if "save_images" in self.tasks and self.labeled_dir:
                        results_to_save.append({"feed_dict": feed_dict, "pred": pred.cpu(), "mask": mask.cpu()})

        if "save_images" in self.tasks and self.labeled_dir:
            saving_config = self.config.get("saving", {})
            save_n_images = saving_config.get("save_n_images")

            if save_n_images and isinstance(save_n_images, int) and save_n_images > 0:
                sampled_results = results_to_save[:save_n_images]
            else:
                sample_ratio = saving_config.get("image_sample_ratio", 1.0)
                num_to_save = int(len(results_to_save) * sample_ratio)
                sampled_results = random.sample(results_to_save, num_to_save) if sample_ratio < 1.0 else results_to_save
            
            print(f"Saving {len(sampled_results)} out of {len(results_to_save)} images...")
            with tqdm(total=len(sampled_results), desc="Stage 2: Saving Images") as t:
                for result in sampled_results:
                    self.save_result_images(result["feed_dict"], result["pred"], result["mask"], self.labeled_dir)
                    t.update()

        return self.summarize_results()

    def save_result_images(self, feed_dict: dict, pred: torch.Tensor, mask: torch.Tensor, save_path: str) -> None:
        """
        단일 배치의 추론 결과를 이미지 파일로 저장합니다.

        Args:
            feed_dict (dict): 데이터로더로부터 받은 원본 데이터 정보.
            pred (torch.Tensor): 모델의 예측 결과 텐서.
            mask (torch.Tensor): 실제 정답 레이블 텐서.
            save_path (str): 이미지를 저장할 디렉토리 경로.
        """
        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
            # Prediction
            p = pred[i].numpy()
            raw_image = np.array(Image.open(image_path).convert("RGB"))
            pred_canvas = get_canvas(raw_image, p, self.data_loader.dataset.class_colors)
            pred_canvas = Image.fromarray(pred_canvas)
            pred_canvas.save(os.path.join(save_path, f"{idx}_pred.png"))

            # Ground Truth
            m = mask[i].numpy()
            gt_canvas = get_canvas(raw_image, m, self.data_loader.dataset.class_colors)
            gt_canvas = Image.fromarray(gt_canvas)
            gt_canvas.save(os.path.join(save_path, f"{idx}_gt.png"))


    def summarize_results(self) -> dict[str, Any]:
        """
        누적된 통계치를 바탕으로 최종 성능 지표를 계산하고, 결과를 JSON 파일로 저장하며,
        마크다운 보고서를 생성합니다.

        Returns:
            dict[str, Any]: 최종 성능 지표 딕셔너리.
        """
        results = {}
        markdown_lines = ["# Evaluation Summary\n"]

        if "calculate_miou" in self.tasks:
            print("\n--- IoU Debug Info ---")
            markdown_lines.append("## IoU Details")
            markdown_lines.append("| Class ID | Class Name | Union | Intersection | IoU |")
            markdown_lines.append("|:---:|:---|:---:|:---:|:---:|")

            class_names = self.data_loader.dataset.classes
            union_sum = self.union_meter.sum.cpu().numpy()
            intersection_sum = self.intersection_meter.sum.cpu().numpy()
            
            low_iou_classes = []
            mid_iou_classes = []
            high_iou_classes = []
            
            ious_all_valid = []

            for i, name in enumerate(class_names):
                iou = 0.0
                if union_sum[i] > 0:
                    iou = intersection_sum[i] / union_sum[i]
                    ious_all_valid.append(iou)

                print(f"  - Class {i:02d} ({name:<25}): Union = {union_sum[i]}, Intersection = {intersection_sum[i]}, IoU = {iou:.4f}")
                markdown_lines.append(f"| {i:02d} | {name} | {union_sum[i]:.1f} | {intersection_sum[i]:.1f} | {iou:.4f} |")
                
                if union_sum[i] > 0:
                    if iou <= 0.1:
                        low_iou_classes.append(name)
                    elif iou <= 0.5:
                        mid_iou_classes.append(name)
                    else: # iou > 0.5
                        high_iou_classes.append(name)

            print("----------------------\n")
            markdown_lines.append("\n## IoU-based Class Groups\n")

            if low_iou_classes:
                print("--- Classes with IoU <= 0.1 ---")
                markdown_lines.append("### Classes with IoU <= 0.1")
                for name in low_iou_classes:
                    print(f"  - {name}")
                    markdown_lines.append(f"- {name}")
                print("---------------------------------\n")
                markdown_lines.append("")

            if mid_iou_classes:
                print("--- Classes with 0.1 < IoU <= 0.5 ---")
                markdown_lines.append("### Classes with 0.1 < IoU <= 0.5")
                for name in mid_iou_classes:
                    print(f"  - {name}")
                    markdown_lines.append(f"- {name}")
                print("-------------------------------------\n")
                markdown_lines.append("")
            
            if high_iou_classes:
                print("--- Classes with IoU > 0.5 ---")
                markdown_lines.append("### Classes with IoU > 0.5")
                for name in high_iou_classes:
                    print(f"  - {name}")
                    markdown_lines.append(f"- {name}")
                print("----------------------------------\n")
                markdown_lines.append("")

            markdown_lines.append("## mIoU Results\n")
            iou_values_over_0_1 = [iou for iou in ious_all_valid if iou > 0.1]
            if iou_values_over_0_1:
                miou_over_0_1 = (sum(iou_values_over_0_1) / len(iou_values_over_0_1)) * 100
                print(f"--- mIoU (IoU > 0.1): {miou_over_0_1:.3f}% ---\n")
                markdown_lines.append(f"- **mIoU (IoU > 0.1):** {miou_over_0_1:.3f}%\n")

            iou_values_over_0_5 = [iou for iou in ious_all_valid if iou > 0.5]
            if iou_values_over_0_5:
                miou_over_0_5 = (sum(iou_values_over_0_5) / len(iou_values_over_0_5)) * 100
                print(f"--- mIoU (IoU > 0.5): {miou_over_0_5:.3f}% ---\n")
                markdown_lines.append(f"- **mIoU (IoU > 0.5):** {miou_over_0_5:.3f}%\n")

            iou_per_class = self.intersection_meter.sum / self.union_meter.sum
            valid_classes_mask = self.union_meter.sum > 0
            
            if valid_classes_mask.sum() > 0:
                miou = iou_per_class[valid_classes_mask].cpu().mean().item() * 100
            else:
                miou = 0.0
            results["mIOU"] = miou
            markdown_lines.append(f"- **Overall mIoU (all valid classes):** {miou:.3f}%\n")

        if "calculate_fps" in self.tasks:
            total_images = self.time_meter.get_count() * self.data_loader.batch_size
            fps = total_images / self.time_meter.sum
            results["fps"] = fps
            markdown_lines.append("## Performance")
            markdown_lines.append(f"- **Inference FPS:** {fps:.2f}")

        if self.run_dir:
            summary_path = os.path.join(self.run_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=4)
            
            report_path = os.path.join(self.run_dir, "evaluation_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(markdown_lines))
            print(f"\nMarkdown report saved to {report_path}")

        return results