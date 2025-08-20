"""
이 파일은 EfficientViT 기반 Semantic Segmentation 모델의 아키텍처를 정의합니다.
- `SegHead`: 백본에서 나온 여러 스케일의 특징(feature)을 융합하여 최종 예측을 만드는 헤드 모듈.
- `EfficientViTSeg`: `EfficientViTBackbone`과 `SegHead`를 결합한 전체 세그멘테이션 모델.
- `efficientvit_seg_b*`, `efficientvit_seg_l*`: 다양한 크기와 데이터셋에 맞춰 사전 정의된
  모델을 생성하는 팩토리 함수들.
"""
from typing import Optional

import torch
import torch.nn as nn

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
)
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTSeg",
    "efficientvit_seg_b0",
    "efficientvit_seg_b1",
    "efficientvit_seg_b2",
    "efficientvit_seg_b3",
    "efficientvit_seg_l1",
    "efficientvit_seg_l2",
]


class SegHead(DAGBlock):
    """
    Semantic Segmentation을 위한 헤드 모듈.

    `DAGBlock`을 상속받아, 백본의 여러 스테이지에서 나온 특징들을 입력으로 받아
    업샘플링과 컨볼루션 연산을 통해 융합하고, 최종적으로 클래스별 세그멘테이션 맵을 출력합니다.
    이는 FPN(Feature Pyramid Network)과 유사한 구조를 가집니다.
    """

    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: Optional[float],
        n_classes: int,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "hswish",
    ):
        """
        Args:
            fid_list (list[str]): 입력으로 사용할 백본 특징들의 이름 리스트 (e.g., ["stage4", "stage3"]).
            in_channel_list (list[int]): 각 백본 특징들의 채널 수 리스트.
            stride_list (list[int]): 각 백본 특징들의 스트라이드(stride) 리스트.
            head_stride (int): 헤드 모듈의 기준 스트라이드. 이 값에 맞춰 다른 특징들이 업샘플링됩니다.
            head_width (int): 헤드 모듈 내부의 기본 채널 수.
            head_depth (int): 헤드 모듈의 중간 블록(middle block) 반복 횟수.
            expand_ratio (float): 중간 블록(MBConv, FusedMBConv)의 확장 비율.
            middle_op (str): 중간 블록으로 사용할 연산 종류 ("mbconv" or "fmbconv").
            final_expand (Optional[float]): 최종 출력 레이어 전 채널 확장 비율.
            n_classes (int): 최종 출력 클래스의 개수.
            dropout (float, optional): 최종 출력 레이어에 적용할 드롭아웃 비율. Defaults to 0.
            norm (str, optional): 사용할 정규화(normalization) 레이어. Defaults to "bn2d".
            act_func (str, optional): 사용할 활성화 함수. Defaults to "hswish".
        """
        inputs = {}
        # 1. 입력 처리: 백본의 각 특징을 받아서 업샘플링 및 채널 수 조절
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                # 스트라이드가 같으면 채널 수만 조절
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                # 스트라이드가 다르면 업샘플링 후 채널 수 조절
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        # 2. 중간 블록: 융합된 특징들을 처리하는 컨볼루션 블록
        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        # 3. 출력 블록: 최종적으로 n_classes개의 채널을 가진 세그멘테이션 맵 생성
        outputs = {
            "segout": OpSequential(
                [
                    (
                        None
                        if final_expand is None
                        else ConvLayer(head_width, int(head_width * final_expand), 1, norm=norm, act_func=act_func)
                    ),
                    ConvLayer(
                        int(head_width * (1 if final_expand is None else final_expand)),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSeg(nn.Module):
    """
    EfficientViT 세그멘테이션 모델.

    백본(`EfficientViTBackbone`)과 세그멘테이션 헤드(`SegHead`)를 결합하여
    end-to-end 세그멘테이션 모델을 구성합니다.
    """

    def __init__(self, backbone: EfficientViTBackbone | EfficientViTLargeBackbone, head: SegHead) -> None:
        """
        Args:
            backbone (EfficientViTBackbone | EfficientViTLargeBackbone): 특징 추출을 위한 백본 네트워크.
            head (SegHead): 특징을 융합하고 최종 예측을 생성하는 헤드 네트워크.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파를 정의합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서. (B, C, H, W)

        Returns:
            torch.Tensor: 모델의 세그멘테이션 예측 결과. (B, n_classes, H', W')
        """
        # 백본을 통해 여러 스케일의 특징을 추출합니다.
        feed_dict = self.backbone(x)
        # 헤드를 통해 특징들을 융합하고 최종 예측을 생성합니다.
        feed_dict = self.head(feed_dict)

        return feed_dict["segout"]


def efficientvit_seg_b0(dataset: str, **kwargs) -> EfficientViTSeg:
    """EfficientViT-B0 백본을 사용하는 세그멘테이션 모델을 생성합니다."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)
    custom_n_classes = kwargs.pop("n_classes", None)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "mapillary":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=124 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError(f"Dataset `{dataset}` is not supported.")
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b1(dataset: str, **kwargs) -> EfficientViTSeg:
    """EfficientViT-B1 백본을 사용하는 세그멘테이션 모델을 생성합니다."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)
    custom_n_classes = kwargs.pop("n_classes", None)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=None,
            n_classes=150 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError(f"Dataset `{dataset}` is not supported.")
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b2(dataset: str, **kwargs) -> EfficientViTSeg:
    """EfficientViT-B2 백본을 사용하는 세그멘테이션 모델을 생성합니다."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)
    custom_n_classes = kwargs.pop("n_classes", None)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=None,
            n_classes=150 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError(f"Dataset `{dataset}` is not supported.")
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b3(dataset: str, **kwargs) -> EfficientViTSeg:
    """EfficientViT-B3 백본을 사용하는 세그멘테이션 모델을 생성합니다."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)
    custom_n_classes = kwargs.pop("n_classes", None)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=None,
            n_classes=150 if custom_n_classes is None else custom_n_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError(f"Dataset `{dataset}` is not supported.")
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_l1(dataset: str, **kwargs) -> EfficientViTSeg:
    """EfficientViT-L1 백본을 사용하는 세그멘테이션 모델을 생성합니다."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)
    custom_n_classes = kwargs.pop("n_classes", None)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=256,
            head_depth=3,
            expand_ratio=1,
            middle_op="fmbconv",
            final_expand=None,
            n_classes=19 if custom_n_classes is None else custom_n_classes,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=8,
            n_classes=150 if custom_n_classes is None else custom_n_classes,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError(f"Dataset `{dataset}` is not supported.")
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_l2(dataset: str, **kwargs) -> EfficientViTSeg:
    """EfficientViT-L2 백본을 사용하는 세그멘테이션 모델을 생성합니다."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)
    custom_n_classes = kwargs.pop("n_classes", None)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=256,
            head_depth=5,
            expand_ratio=1,
            middle_op="fmbconv",
            final_expand=None,
            n_classes=19 if custom_n_classes is None else custom_n_classes,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=8,
            n_classes=150 if custom_n_classes is None else custom_n_classes,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError(f"Dataset `{dataset}` is not supported.")
    model = EfficientViTSeg(backbone, head)
    return model
