
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

def _fast_hist(label_true, label_pred, n_class):
    """
    혼동 행렬(confusion matrix)을 빠르게 계산합니다.

    Args:
        label_true (ndarray): 실제 레이블.
        label_pred (ndarray): 예측 레이블.
        n_class (int): 클래스의 수.

    Returns:
        ndarray: 계산된 혼동 행렬.
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

class SegTrainer(Trainer):
    """
    Semantic Segmentation을 위한 Trainer 클래스.
    `Trainer`를 상속받아 세그멘테이션 태스크에 맞는 학습 및 검증 로직을 구현합니다.
    """
    def __init__(self, path: str, model: nn.Module, data_provider):
        """
        SegTrainer를 초기화합니다.

        Args:
            path (str): 실험 결과를 저장할 경로.
            model (nn.Module): 학습할 모델.
            data_provider: 데이터 프로바이더 객체.
        """
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )
        # 손실 함수로 CrossEntropyLoss를 사용하며, 255는 무시할 인덱스로 설정합니다.
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        # 데이터셋의 클래스 수를 data_provider에서 가져옵니다.
        self.n_classes = data_provider.n_classes

    def before_step(self, sample: dict[str, Any]) -> dict[str, Any]:
        img: torch.Tensor = sample["image"]
        lbl: torch.Tensor = sample["label"]
        img = img.to("cuda", non_blocking=True, memory_format=torch.channels_last)
        lbl = lbl.to("cuda", non_blocking=True)
        return {"image": img, "label": lbl}

    def _validate(self, model, data_loader, epoch) -> dict[str, Any]:
        """검증 데이터셋으로 모델 성능을 평가합니다."""
        val_loss = AverageMeter() # 손실 평균을 계산하기 위한 객체
        val_miou = AverageMeter() # mIoU 평균을 계산하기 위한 객체
        hist = np.zeros((self.n_classes, self.n_classes)) # 혼동 행렬 초기화

        with torch.no_grad(): # 그래디언트 계산 비활성화
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Epoch #{epoch + 1}",
                disable=not is_master(),
                file=sys.stdout,
            ) as t:
                for sample in data_loader:
                    feed_dict = self.before_step(sample)
                    images = feed_dict["image"]
                    labels = feed_dict["label"]

                    # 모델 예측
                    output = model(images)
                    output = F.interpolate(output, size=labels.shape[1:], mode='bilinear', align_corners=True)

                    # 손실 계산
                    loss = self.criterion(output, labels)

                    val_loss.update(loss.item(), images.size(0))
                    
                    # IoU 계산을 위한 예측 및 혼동 행렬 업데이트
                    pred = torch.argmax(output, dim=1)
                    hist += _fast_hist(labels.cpu().numpy(), pred.cpu().numpy(), self.n_classes)

                    t.set_postfix(
                        {
                            "loss": val_loss.avg,
                            "#samples": val_loss.get_count(),
                            "bs": images.shape[0],
                            "res": images.shape[2],
                        }
                    )
                    t.update()
        
        # 분산 학습 환경에서 모든 프로세스의 혼동 행렬을 동기화합니다.
        if self.data_provider.num_replicas is not None:
            hist_tensor = torch.from_numpy(hist).cuda()
            hist = sync_tensor(hist_tensor, reduce='sum').cpu().numpy()

        # mIoU 계산
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        miou = np.nanmean(iu)
        val_miou.update(miou, val_loss.get_count())

        return {"val_loss": val_loss.avg, "val_miou": val_miou.avg}

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        """단일 학습 스텝(배치)을 실행합니다."""
        images = feed_dict['image']
        labels = feed_dict['label']

        # Automatic Mixed Precision (AMP) 활성화
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            output = self.model(images)
            output = F.interpolate(output, size=labels.shape[1:], mode='bilinear', align_corners=True)
            loss = self.criterion(output, labels)
        
        # 손실에 대한 그래디언트를 계산하고 역전파합니다.
        self.scaler.scale(loss).backward()
        return {"loss": loss}

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        """한 에폭 동안의 전체 학습 루프를 실행합니다."""
        train_loss = AverageMeter()

        with tqdm(
            total=len(self.data_provider.train),
            desc=f"Train Epoch #{epoch + 1}",
            disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for sample in self.data_provider.train:
                self.optimizer.zero_grad() # 그래디언트 초기화
                feed_dict = self.before_step(sample)
                output_dict = self.run_step(feed_dict) # 학습 스텝 실행
                self.after_step() # 옵티마이저 및 스케줄러 업데이트

                train_loss.update(output_dict["loss"].item(), sample['image'].size(0))

                # 진행 상황 표시
                t.set_postfix(
                    {
                        "loss": train_loss.avg,
                        "bs": sample['image'].shape[0],
                        "res": sample['image'].shape[2],
                        "lr": list_join(
                            sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                            "#",
                            "%.1E",
                        ),
                        "progress": self.run_config.progress,
                    }
                )
                t.update()
        return {"train_loss": train_loss.avg}

    def train(self) -> None:
        """전체 학습 과정을 관리하고 실행합니다."""
        # Early stopping 설정
        early_stopping_patience = getattr(self.run_config, 'early_stopping_patience', 0)
        early_stopping_metric = getattr(self.run_config, 'early_stopping_metric', 'val_miou')
        early_stopping_counter = 0
        
        if early_stopping_metric == 'val_miou':
            best_val_for_early_stopping = self.best_val
            is_better = lambda current, best: current > best
        elif early_stopping_metric == 'val_loss':
            best_val_for_early_stopping = float('inf')
            is_better = lambda current, best: current < best
        else:
            # 지원하지 않는 메트릭이면 early stopping 비활성화
            early_stopping_patience = 0

        if early_stopping_patience > 0 and is_master():
            self.write_log(f"Early stopping enabled: patience={early_stopping_patience}, metric='{early_stopping_metric}'")

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            self.data_provider.set_epoch(epoch)
            self.train_one_epoch(epoch)
            val_info_dict = self.validate(epoch=epoch, is_test=False)
            self.write_log(f"Epoch {epoch + 1} Validation: Loss={val_info_dict['val_loss']:.4f}, mIoU={val_info_dict['val_miou']:.4f}", prefix="valid")

            # 최고의 성능을 보인 모델을 저장합니다. (기존 로직)
            is_best = val_info_dict["val_miou"] > self.best_val
            self.best_val = max(val_info_dict["val_miou"], self.best_val)

            self.save_model(
                only_state_dict=True,
                epoch=epoch,
                model_name="model_best.pt" if is_best else "checkpoint.pt",
            )

            # Early stopping 로직
            if early_stopping_patience > 0:
                current_metric_val = val_info_dict[early_stopping_metric]
                if is_better(current_metric_val, best_val_for_early_stopping):
                    best_val_for_early_stopping = current_metric_val
                    early_stopping_counter = 0
                    if is_master():
                        self.write_log(f"Early stopping metric '{early_stopping_metric}' improved to {best_val_for_early_stopping:.4f}.", prefix="valid")
                else:
                    early_stopping_counter += 1
                    if is_master():
                        self.write_log(
                            f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}. "
                            f"Metric '{early_stopping_metric}' did not improve from {best_val_for_early_stopping:.4f}.",
                            prefix="valid"
                        )
                
                if early_stopping_counter >= early_stopping_patience:
                    if is_master():
                        self.write_log(f"Early stopping triggered after {epoch + 1} epochs.", prefix="valid")
                    break
