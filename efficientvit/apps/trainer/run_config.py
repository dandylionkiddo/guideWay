"""
이 파일은 학습 실행과 관련된 모든 하이퍼파라미터와 설정을 관리하는
`RunConfig` 클래스를 정의합니다.

YAML 설정 파일의 `run_config` 섹션에 해당하는 값들을 객체 속성으로 가지며,
옵티마이저 및 학습률 스케줄러 생성, 학습 진행 상태 관리 등의 기능을 제공합니다.
"""
import json
from typing import Any

import numpy as np
import torch.nn as nn

from efficientvit.apps.utils import CosineLRwithWarmup, build_optimizer

__all__ = ["Scheduler", "RunConfig"]


class Scheduler:
    """
    학습 진행률을 전역적으로 추적하기 위한 간단한 클래스.
    `RunConfig`의 `progress` 속성 값을 저장하는 데 사용됩니다.
    """
    PROGRESS = 0


class RunConfig:
    """
    학습 실행에 필요한 모든 설정을 담는 데이터 클래스.
    YAML 파일로부터 동적으로 속성이 설정됩니다.
    """

    # 클래스 변수로 기대하는 속성들을 타입 어노테이션과 함께 정의합니다.
    # `__init__`에서 이 어노테이션을 사용하여 필요한 모든 설정값이 주어졌는지 확인합니다.
    n_epochs: int
    init_lr: float
    warmup_epochs: int
    warmup_lr: float
    lr_schedule_name: str
    lr_schedule_param: dict
    optimizer_name: str
    optimizer_params: dict
    weight_decay: float
    no_wd_keys: list
    grad_clip: float  # None을 허용하여 그래디언트 클리핑 비활성화
    reset_bn: bool
    reset_bn_size: int
    reset_bn_batch_size: int
    eval_image_size: list  # None을 허용하여 data_provider의 image_size 사용

    @property
    def none_allowed(self) -> list[str]:
        """None 값을 허용하는 속성들의 리스트를 반환합니다."""
        return ["grad_clip", "eval_image_size"]

    def __init__(self, **kwargs: Any) -> None:
        """
        YAML 파일 등에서 받은 키워드 인자들을 객체의 속성으로 설정합니다.

        Args:
            **kwargs (Any): `run_config` 섹션의 모든 키-값 쌍.
        """
        for k, val in kwargs.items():
            setattr(self, k, val)

        # 클래스에 정의된 모든 타입 어노테이션을 확인하여,
        # 필요한 모든 설정값이 제대로 주어졌는지 검증합니다.
        annotations = {}
        for cls in type(self).mro():
            if hasattr(cls, "__annotations__"):
                annotations.update(cls.__annotations__)
        for k, k_type in annotations.items():
            assert hasattr(self, k), f"Key {k} with type {k_type} is required for initialization."
            attr = getattr(self, k)
            # `none_allowed`에 포함된 키는 None 타입도 허용합니다.
            if k in self.none_allowed:
                k_type = (k_type, type(None))
            assert isinstance(attr, k_type), f"Key {k} must be of type {k_type}, but got {type(attr)}."

        self.global_step = 0
        self.batch_per_epoch = 1

    def build_optimizer(self, network: nn.Module) -> tuple[Any, Any]:
        """
        설정값에 따라 옵티마이저와 학습률(LR) 스케줄러를 생성합니다.

        `no_wd_keys`에 지정된 파라미터 그룹에 대해서는 weight decay를 적용하지 않는
        차등적인 옵티마이저 설정이 가능합니다.

        Args:
            network (nn.Module): 옵티마이저가 최적화할 모델.

        Returns:
            tuple[Any, Any]: 생성된 옵티마이저와 LR 스케줄러.
        """
        param_groups = []
        # 파라미터 이름에 `no_wd_keys`의 문자열이 포함되는지에 따라 weight decay 적용 여부를 결정
        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            
            weight_decay = self.weight_decay
            if self.no_wd_keys and any(key in name for key in self.no_wd_keys):
                weight_decay = 0
            
            param_groups.append({"params": [param], "weight_decay": weight_decay})

        optimizer = build_optimizer(param_groups, self.optimizer_name, self.optimizer_params, self.init_lr)

        # 코사인 학습률 스케줄러 (웜업 포함) 생성
        if self.lr_schedule_name == "cosine":
            decay_steps = [epoch * self.batch_per_epoch for epoch in self.lr_schedule_param.get("step", [])]
            decay_steps.append(self.n_epochs * self.batch_per_epoch)
            decay_steps.sort()
            
            lr_scheduler = CosineLRwithWarmup(
                optimizer,
                warmup_steps=self.warmup_epochs * self.batch_per_epoch,
                warmup_lr=self.warmup_lr,
                decay_steps=decay_steps,
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_schedule_name} is not implemented.")
        
        return optimizer, lr_scheduler

    def update_global_step(self, epoch: int, batch_id: int = 0) -> None:
        """전역 스텝 카운터를 업데이트합니다."""
        self.global_step = epoch * self.batch_per_epoch + batch_id
        Scheduler.PROGRESS = self.progress

    @property
    def progress(self) -> float:
        """전체 학습 과정 대비 현재 진행률(0.0 ~ 1.0)을 계산합니다."""
        warmup_steps = self.warmup_epochs * self.batch_per_epoch
        total_steps = self.n_epochs * self.batch_per_epoch
        current_steps = max(0, self.global_step - warmup_steps)
        return current_steps / total_steps

    def step(self) -> None:
        """전역 스텝을 1 증가시킵니다."""
        self.global_step += 1
        Scheduler.PROGRESS = self.progress

    def get_remaining_epoch(self, epoch: int, post: bool = True) -> int:
        """남은 에폭 수를 계산합니다."""
        return self.n_epochs + self.warmup_epochs - epoch - int(post)

    def epoch_format(self, epoch: int) -> str:
        """로깅을 위한 에폭 문자열 포맷을 생성합니다. (예: [ 1/100])"""
        epoch_format = f"%.{len(str(self.n_epochs))}d"
        epoch_format = f"[{epoch_format}/{epoch_format}]"
        # 웜업 에폭을 제외하고 현재 에폭과 전체 에폭을 표시
        current_epoch_display = epoch + 1 - self.warmup_epochs
        return epoch_format % (current_epoch_display, self.n_epochs)
