"""
이 파일은 EfficientViT 모델의 백본(backbone) 아키텍처를 정의합니다.
백본은 입력 이미지로부터 계층적인 특징(hierarchical features)을 추출하는 역할을 합니다.

- `EfficientViTBackbone`: EfficientViT의 B0-B3 버전을 위한 백본.
- `EfficientViTLargeBackbone`: EfficientViT의 L0-L3 버전을 위한 백본.
- `efficientvit_backbone_*`: 다양한 크기의 백본 모델을 생성하는 팩토리 함수.
"""
from typing import Optional

import torch
import torch.nn as nn

from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]


class EfficientViTBackbone(nn.Module):
    """
    EfficientViT (B0-B3) 모델의 백본 네트워크.

    입력 이미지로부터 여러 단계(stage)에 걸쳐 특징을 추출합니다.
    초기 스테이지는 주로 컨볼루션 블록(MBConv)으로 구성되고,
    후기 스테이지는 `EfficientViTBlock`(어텐션 기반)을 포함합니다.
    """

    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels: int = 3,
        dim: int = 32,
        expand_ratio: float = 4,
        norm: str = "bn2d",
        act_func: str = "hswish",
    ) -> None:
        """
        Args:
            width_list (list[int]): 각 스테이지의 출력 채널 수 리스트.
            depth_list (list[int]): 각 스테이지의 블록 반복 횟수 리스트.
            in_channels (int, optional): 입력 이미지의 채널 수. Defaults to 3.
            dim (int, optional): `EfficientViTBlock` 내부의 어텐션 차원. Defaults to 32.
            expand_ratio (float, optional): MBConv 블록의 확장 비율. Defaults to 4.
            norm (str, optional): 사용할 정규화 레이어. Defaults to "bn2d".
            act_func (str, optional): 사용할 활성화 함수. Defaults to "hswish".
        """
        super().__init__()

        self.width_list = []
        
        # 1. Input Stem: 초기 특징 추출 단계
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # 2. Stages: 점진적으로 특징을 추출하고 다운샘플링하는 단계들
        self.stages = []
        # Stage 1, 2: Local feature extraction (MBConv)
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        # Stage 3, 4: Global feature extraction (EfficientViTBlock)
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            # 첫 블록은 다운샘플링을 위한 MBConv
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w
            # 나머지 블록은 어텐션 기반의 EfficientViTBlock
            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """MBConv 또는 DSConv 블록을 생성하는 정적 메서드."""
        if expand_ratio == 1:
            # 확장 비율이 1이면 Depthwise-Separable Convolution 사용
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            # 확장 비율이 1보다 크면 MobileNetV2의 Inverted Residual Block (MBConv) 사용
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        백본의 순전파를 정의합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서.

        Returns:
            dict[str, torch.Tensor]: 각 스테이지의 출력을 담은 딕셔너리.
                                     세그멘테이션 헤드 등에서 멀티스케일 특징을 활용하기 위함입니다.
        """
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict[f"stage{stage_id}"] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    """EfficientViT-B0 백본 모델을 생성합니다."""
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    """EfficientViT-B1 백본 모델을 생성합니다."""
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    """EfficientViT-B2 백본 모델을 생성합니다."""
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    """EfficientViT-B3 백본 모델을 생성합니다."""
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    """
    EfficientViT Large (L0-L3) 모델의 백본 네트워크.

    기본 백본보다 더 다양한 블록 조합(ResBlock, FusedMBConv 등)을 사용하여
    더 큰 모델을 구성할 수 있도록 설계되었습니다.
    """

    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: Optional[list[str]] = None,
        expand_list: Optional[list[float]] = None,
        fewer_norm_list: Optional[list[bool]] = None,
        in_channels: int = 3,
        qkv_dim: int = 32,
        norm: str = "bn2d",
        act_func: str = "gelu",
    ) -> None:
        """
        Args:
            width_list (list[int]): 각 스테이지의 출력 채널 수 리스트.
            depth_list (list[int]): 각 스테이지의 블록 반복 횟수 리스트.
            block_list (Optional[list[str]], optional): 각 스테이지에서 사용할 블록 타입 리스트. Defaults to ["res", "fmb", "fmb", "mb", "att"].
            expand_list (Optional[list[float]], optional): 각 스테이지의 확장 비율 리스트. Defaults to [1, 4, 4, 4, 6].
            fewer_norm_list (Optional[list[bool]], optional): 각 스테이지에서 정규화 레이어를 더 적게 사용할지 여부. Defaults to [False, False, False, True, True].
            in_channels (int, optional): 입력 채널 수. Defaults to 3.
            qkv_dim (int, optional): 어텐션 블록의 QKV 차원. Defaults to 32.
            norm (str, optional): 사용할 정규화 레이어. Defaults to "bn2d".
            act_func (str, optional): 사용할 활성화 함수. Defaults to "gelu".
        """
        super().__init__()
        block_list = ["res", "fmb", "fmb", "mb", "att"] if block_list is None else block_list
        expand_list = [1, 4, 4, 4, 6] if expand_list is None else expand_list
        fewer_norm_list = [False, False, False, True, True] if fewer_norm_list is None else fewer_norm_list

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """다양한 타입의 로컬 블록(res, fmb, mb)을 생성하는 정적 메서드."""
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        백본의 순전파를 정의합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서.

        Returns:
            dict[str, torch.Tensor]: 각 스테이지의 출력을 담은 딕셔너리.
        """
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict[f"stage{stage_id}"] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    """EfficientViT-L0 백본 모델을 생성합니다."""
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    """EfficientViT-L1 백본 모델을 생성합니다."""
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    """EfficientViT-L2 백본 모델을 생성합니다."""
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    """EfficientViT-L3 백본 모델을 생성합니다."""
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone
