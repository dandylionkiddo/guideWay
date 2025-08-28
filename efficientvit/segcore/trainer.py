
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np

from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter, is_master, sync_tensor
from efficientvit.models.utils import list_join

__all__ = ["SegTrainer"]


def _fast_hist(label_true: np.ndarray, label_pred: np.ndarray, n_class: int) -> np.ndarray:
    """
    혼동 행렬(confusion matrix)을 효율적으로 계산합니다.

    Args:
        label_true (np.ndarray): 실제 레이블 값으로 이루어진 배열.
        label_pred (np.ndarray): 모델이 예측한 레이블 값으로 이루어진 배열.
        n_class (int): 전체 클래스의 개수.

    Returns:
        np.ndarray: (n_class, n_class) 형태의 혼동 행렬.
    """
    # 유효한 레이블 범위 (0 <= label < n_class) 에 해당하는 마스크를 생성합니다.
    mask = (label_true >= 0) & (label_true < n_class)
    # 혼동 행렬을 1차원 배열로 계산한 뒤, (n_class, n_class) 형태로 변환합니다.
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class**2,
    ).reshape(n_class, n_class)
    return hist


class SegTrainer(Trainer):
    """
    Semantic Segmentation을 위한 Trainer 클래스.

    `Trainer` 베이스 클래스를 상속받아, 세그멘테이션 태스크에 특화된
    학습 및 검증 로직을 구현합니다.
    """

    def __init__(self, path: str, model: nn.Module, data_provider: Any) -> None:
        """
        SegTrainer를 초기화합니다.

        Args:
            path (str): 실험 결과(로그, 체크포인트)를 저장할 경로.
            model (nn.Module): 학습 및 검증에 사용할 모델.
            data_provider (Any): 학습 및 검증 데이터를 제공하는 데이터 프로바이더 객체.
        """
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )
        self.n_classes = data_provider.n_classes

    def prep_for_training(self, run_config: Any, ema_decay: float | None = None, amp: str = "fp32") -> None:
        super().prep_for_training(run_config, ema_decay, amp)

        # Loss 함수를 설정합니다.
        loss_config = self.run_config.loss
        if loss_config['name'] == "focal":
            from efficientvit.models.nn.loss import FocalLoss

            self.criterion = FocalLoss(
                alpha=loss_config.get("focal_loss_alpha", 0.25),
                gamma=loss_config.get("focal_loss_gamma", 2.0),
                ignore_index=255,
            )
            if is_master():
                self.write_log("Using Focal Loss")
        elif loss_config['name'] == "dice":
            from efficientvit.models.nn.loss import DiceLoss

            self.criterion = DiceLoss(n_classes=self.n_classes)
            if is_master():
                self.write_log("Using Dice Loss")
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
            if is_master():
                self.write_log("Using Cross Entropy Loss")

    def before_step(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        학습/검증 스텝이 실행되기 전, 데이터를 GPU로 이동시키는 전처리 작업을 수행합니다.

        Args:
            sample (dict[str, Any]): 데이터 프로바이더로부터 받은 데이터 샘플.
                                     {'image': torch.Tensor, 'label': torch.Tensor} 형태를 기대합니다.

        Returns:
            dict[str, torch.Tensor]: 데이터가 GPU로 이동된 딕셔너리.
        """
        img: torch.Tensor = sample["image"]
        lbl: torch.Tensor = sample["label"]
        # `non_blocking=True`는 비동기 전송을 가능하게 하여 성능을 향상시킬 수 있습니다.
        # `memory_format=torch.channels_last`는 특정 연산(예: Conv)의 성능을 향상시킬 수 있습니다.
        img = img.to("cuda", non_blocking=True, memory_format=torch.channels_last)
        lbl = lbl.to("cuda", non_blocking=True)
        return {"image": img, "label": lbl}

    def _validate(self, model: nn.Module, data_loader: Any, epoch: int) -> dict[str, float]:
        """
        검증 데이터셋을 사용하여 모델의 성능을 평가합니다.

        Args:
            model (nn.Module): 평가할 모델.
            data_loader (Any): 검증 데이터로더.
            epoch (int): 현재 에폭 번호 (로깅용).

        Returns:
            dict[str, float]: 검증 결과 (손실, mIoU, 클래스별 상세 지표)를 담은 딕셔너리.
        """
        val_loss = AverageMeter()
        hist = np.zeros((self.n_classes, self.n_classes))

        model.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():  # 그래디언트 계산 비활성화
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Epoch #{epoch + 1}",
                disable=not is_master(),
                file=sys.stdout,
            ) as t:
                for sample in data_loader:
                    feed_dict = self.before_step(sample)
                    images, labels = feed_dict["image"], feed_dict["label"]

                    output = model(images)
                    output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=True)

                    loss = self.criterion(output, labels)
                    val_loss.update(loss.item(), images.size(0))

                    pred = torch.argmax(output, dim=1)
                    hist += _fast_hist(labels.cpu().numpy(), pred.cpu().numpy(), self.n_classes)

                    t.set_postfix({"loss": val_loss.avg, "bs": images.shape[0]})
                    t.update()

        if self.data_provider.num_replicas is not None:
            hist_tensor = torch.from_numpy(hist).cuda()
            hist = sync_tensor(hist_tensor, reduce="sum").cpu().numpy()

        # 클래스별 intersection, union, IoU 계산
        intersection = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        # 0으로 나누는 경우를 방지하기 위해 작은 값(epsilon)을 더합니다.
        iu = intersection / (union + 1e-10)
        miou = np.nanmean(iu)

        # 클래스 이름 가져오기 (data_provider에 있다고 가정)
        try:
            class_names = self.data_provider.classes
        except AttributeError:
            class_names = [f"class_{i}" for i in range(self.n_classes)]

        # 결과 로깅
        if is_master():
            self.write_log("-" * 80, prefix="valid")
            self.write_log(f"{'Class':<20} | {'IoU':>10} | {'Intersection':>15} | {'Union':>15}", prefix="valid")
            self.write_log("-" * 80, prefix="valid")
            for i, class_name in enumerate(class_names):
                self.write_log(
                    f"{class_name:<20} | {iu[i]:>10.4f} | {int(intersection[i]):>15} | {int(union[i]):>15}",
                    prefix="valid",
                )
            self.write_log("-" * 80, prefix="valid")

        # 결과 딕셔너리 구성
        results = {
            "val_loss": val_loss.avg,
            "val_miou": miou,
        }
        # 클래스별 상세 결과 추가
        for i, class_name in enumerate(class_names):
            results[f"val_iou_{class_name}"] = iu[i]
            results[f"val_intersection_{class_name}"] = intersection[i]
            results[f"val_union_{class_name}"] = union[i]

        return results

    def run_step(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        단일 학습 스텝(배치)을 실행하고 손실을 계산합니다.

        Args:
            feed_dict (dict[str, torch.Tensor]): `before_step`을 거친 GPU 텐서 딕셔너리.

        Returns:
            dict[str, torch.Tensor]: 계산된 손실 값을 담은 딕셔너리. {'loss': torch.Tensor}
        """
        images = feed_dict["image"]
        labels = feed_dict["label"]

        # AMP(Automatic Mixed Precision)를 사용하여 학습 속도를 높이고 메모리 사용량을 줄입니다.
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            output = self.model(images)
            output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=True)
            loss = self.criterion(output, labels)

        # 스케일된 손실에 대해 역전파를 수행합니다.
        self.scaler.scale(loss).backward()
        return {"loss": loss}

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """
        한 에폭(epoch) 동안의 전체 학습 루프를 실행합니다.

        Args:
            epoch (int): 현재 에폭 번호.

        Returns:
            dict[str, float]: 에폭의 평균 학습 손실. {'train_loss': float}
        """
        self.model.train()  # 모델을 학습 모드로 설정
        train_loss = AverageMeter()

        with tqdm(
            total=len(self.data_provider.train),
            desc=f"Train Epoch #{epoch + 1}",
            disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for sample in self.data_provider.train:
                self.optimizer.zero_grad()
                feed_dict = self.before_step(sample)
                output_dict = self.run_step(feed_dict)
                self.after_step()  # 옵티마이저 스텝 및 LR 스케줄러 스텝

                train_loss.update(output_dict["loss"].item(), sample["image"].size(0))

                t.set_postfix(
                    {
                        "loss": train_loss.avg,
                        "lr": f'{self.optimizer.param_groups[0]["lr"]:.1E}',
                    }
                )
                t.update()
        return {"train_loss": train_loss.avg}

    def train(self) -> None:
        """
        전체 학습 과정을 관리하고 실행합니다.
        `start_epoch`부터 `n_epochs`까지 루프를 돌며 학습과 검증을 반복하고,
        모델을 저장하며, 조기 종료(early stopping) 로직을 처리합니다.
        """
        # 조기 종료(Early stopping) 관련 설정을 run_config에서 가져옵니다.
        early_stopping_patience = getattr(self.run_config, "early_stopping_patience", 0)
        early_stopping_metric = getattr(self.run_config, "early_stopping_metric", "val_miou")
        early_stopping_counter = 0

        if early_stopping_metric == "val_miou":
            best_val_for_early_stopping = self.best_val
            is_better = lambda current, best: current > best
        elif early_stopping_metric == "val_loss":
            best_val_for_early_stopping = float("inf")
            is_better = lambda current, best: current < best
        else:
            early_stopping_patience = 0  # 지원하지 않는 메트릭이면 조기 종료 비활성화

        if early_stopping_patience > 0 and is_master():
            self.write_log(f"Early stopping enabled: patience={early_stopping_patience}, metric='{early_stopping_metric}'")

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            self.data_provider.set_epoch(epoch)
            self.train_one_epoch(epoch)
            val_info_dict = self.validate(epoch=epoch, is_test=False)

            val_loss = val_info_dict["val_loss"]
            val_miou = val_info_dict["val_miou"]
            self.write_log(f"Epoch {epoch + 1} Validation: Loss={val_loss:.4f}, mIoU={val_miou:.4f}", prefix="valid")

            # mIoU 기준으로 최고의 모델을 저장합니다.
            is_best = val_miou > self.best_val
            self.best_val = max(val_miou, self.best_val)
            self.save_model(
                only_state_dict=True,
                epoch=epoch,
                model_name="model_best.pt" if is_best else "checkpoint.pt",
            )

            # 조기 종료 로직을 수행합니다.
            if early_stopping_patience > 0:
                current_metric_val = val_info_dict[early_stopping_metric]
                if is_better(current_metric_val, best_val_for_early_stopping):
                    best_val_for_early_stopping = current_metric_val
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping_patience:
                    if is_master():
                        self.write_log(f"Early stopping triggered after {epoch + 1} epochs.", prefix="valid")
                    break
