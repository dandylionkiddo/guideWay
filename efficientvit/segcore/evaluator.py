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
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        # Create a mask to ignore pixels with the specified ignore_index.
        # The mask has `True` for valid pixels and `False` for ignored pixels.
        mask = targets != self.ignore_index

        # --- Union Calculation ---
        # To calculate the union, we count the number of pixels for each class
        # in both the predictions (outputs) and the ground truth (targets) for valid pixels.

        # torch.histc requires a float tensor. We cast the integer class indices to float.
        # We add 1 to class indices because histc's range is [min, max], and class 0 would be missed if min=1.
        outputs_float = (outputs + 1).to(torch.float32)
        targets_float = (targets + 1).to(torch.float32)

        # Count pixels for each class in the model's predictions, ignoring pixels based on the mask.
        outputs_hist = torch.histc(
            outputs_float * mask,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )

        # Count pixels for each class in the ground truth labels, ignoring pixels based on the mask.
        targets_hist = torch.histc(
            targets_float * mask,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )

        # --- Intersection Calculation ---
        # The intersection is the set of pixels where the prediction and the ground truth are the same.

        # Create a boolean tensor indicating where the prediction matches the target on valid pixels.
        intersection_tensor = (outputs == targets) * mask

        # Get the class values for the intersecting (correctly predicted) pixels.
        # We use the same float-casted, incremented targets tensor and apply the intersection mask.
        intersection_values = targets_float * intersection_tensor

        # Count the number of intersecting pixels for each class.
        intersections_hist = torch.histc(
            intersection_values,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )

        # --- Final Metrics ---
        # The union is the sum of pixels in both sets minus the intersection.
        # A_union_B = A + B - A_intersection_B
        unions = outputs_hist + targets_hist - intersections_hist

        return {
            "i": intersections_hist,
            "u": unions,
        }


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple | list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        if k < len(colors):
            seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


class Evaluator:
    def __init__(self, model: torch.nn.Module, data_loader: DataLoader, config: dict) -> None:
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.tasks = config["evaluation_tasks"]

        # init metrics
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
        base_save_path = self.config.get("save_path")
        if base_save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(base_save_path, timestamp)
            self.labeled_dir = os.path.join(self.run_dir, "labeled")
            os.makedirs(self.labeled_dir, exist_ok=True)
            print(f"Saving results to {self.run_dir}")

    def evaluate(self) -> dict[str, Any]:
        self._setup_save_dirs()

        # Stage 1: Inference and Metrics Calculation
        results_to_save = []
        with torch.inference_mode():
            with tqdm(total=len(self.data_loader), desc=f"Stage 1: Inference") as t:
                for feed_dict in self.data_loader:
                    images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()

                    start_time = time.time()
                    output = self.model(images)
                    self.time_meter.update(time.time() - start_time)

                    if output.shape[-2:] != mask.shape[-2:]:
                        output = resize(output, size=mask.shape[-2:])

                    pred = torch.argmax(output, dim=1)

                    if "calculate_miou" in self.tasks:
                        stats = self.iou_metric(pred, mask)
                        self.intersection_meter.update(stats["i"])
                        self.union_meter.update(stats["u"])
                        # Calculate IoU per class
                        iou_per_class = self.intersection_meter.sum / self.union_meter.sum

                        # Filter out classes where union is 0
                        valid_classes_mask = self.union_meter.sum > 0

                        # Calculate mIoU only for valid classes
                        if valid_classes_mask.sum() > 0:
                            miou = iou_per_class[valid_classes_mask].cpu().mean().item() * 100
                        else:
                            miou = 0.0 # No valid classes yet, or all unions are zero

                        t.set_postfix({"mIOU": f"{miou:.2f}%"})

                    t.update()

                    if "save_images" in self.tasks and self.labeled_dir:
                        results_to_save.append(
                            {
                                "feed_dict": feed_dict,
                                "pred": pred.cpu(),
                            }
                        )

        # Stage 2: Saving result images
        if "save_images" in self.tasks and self.labeled_dir:
            # Get sampling ratio from config, default to 1.0 (save all)
            saving_config = self.config.get("saving", {})
            sample_ratio = saving_config.get("image_sample_ratio", 1.0)

            if sample_ratio >= 1.0:
                sampled_results = results_to_save
                print(f"Saving all {len(results_to_save)} images...")
            else:
                num_to_save = int(len(results_to_save) * sample_ratio)
                # Ensure we save at least one image if ratio > 0
                if num_to_save == 0 and len(results_to_save) > 0 and sample_ratio > 0:
                    num_to_save = 1
                
                print(f"Randomly saving {num_to_save} out of {len(results_to_save)} images ({sample_ratio:.0%})...")
                sampled_results = random.sample(results_to_save, num_to_save)
            
            with tqdm(total=len(sampled_results), desc="Stage 2: Saving Images") as t:
                for result in sampled_results:
                    self.save_result_images(result["feed_dict"], result["pred"], self.labeled_dir)
                    t.update()

        return self.summarize_results()

    def save_result_images(self, feed_dict: dict, pred: torch.Tensor, save_path: str) -> None:
        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
            p = pred[i].numpy()
            raw_image = np.array(Image.open(image_path).convert("RGB"))
            canvas = get_canvas(raw_image, p, self.data_loader.dataset.class_colors)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(save_path, f"{idx}.png"))

    def summarize_results(self) -> dict[str, Any]:
        results = {}
        if "calculate_miou" in self.tasks:
            # --- DEBUGGING CODE START ---
            print("\n--- IoU Debug Info ---")
            # 클래스 이름과 함께 Union 및 Intersection 값을 출력
            class_names = self.data_loader.dataset.classes
            union_sum = self.union_meter.sum.cpu().numpy()
            intersection_sum = self.intersection_meter.sum.cpu().numpy()
            for i, name in enumerate(class_names):
                print(f"  - Class {i:02d} ({name:<25}): Union = {union_sum[i]}, Intersection = {intersection_sum[i]}")
            print("----------------------\n")
            # --- DEBUGGING CODE END ---

            # Calculate IoU for each class
            iou_per_class = self.intersection_meter.sum / self.union_meter.sum

            # Filter out classes where union is 0 (i.e., class not present in GT or prediction)
            # This avoids division by zero and ensures only relevant classes contribute to mIoU
            valid_classes_mask = self.union_meter.sum > 0
            
            # Calculate mIoU only for valid classes
            if valid_classes_mask.sum() > 0: # Ensure there's at least one valid class
                miou = iou_per_class[valid_classes_mask].cpu().mean().item() * 100
            else:
                miou = 0.0 # No valid classes to calculate mIoU
            results["mIOU"] = miou

        if "calculate_fps" in self.tasks:
            total_images = len(self.data_loader.dataset)
            total_inference_time = self.time_meter.sum
            fps = total_images / total_inference_time
            results["fps"] = fps

        if self.run_dir:
            summary_path = os.path.join(self.run_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=4)

        return results
