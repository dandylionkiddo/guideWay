"""
이 파일은 "모델 동물원(Model Zoo)" 역할을 합니다.
사전 정의된 EfficientViT 세그멘테이션 모델들을 등록하고, 사용자가 모델 이름만으로
쉽게 모델 인스턴스를 생성하고 사전 학습된 가중치까지 로드할 수 있도록
`create_efficientvit_seg_model` 팩토리 함수를 제공합니다.
"""
from typing import Callable, Optional

import torch.nn as nn

from efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
    efficientvit_seg_l1,
    efficientvit_seg_l2,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file
from efficientvit.models.segformer.model import (
    segformer_b0,
    segformer_b1,
    segformer_b2,
    segformer_b3,
    segformer_b4,
    segformer_b5,
)


__all__ = ["create_efficientvit_seg_model", "create_segformer_model"]


# 등록된 EfficientViT 세그멘테이션 모델 정보를 담고 있는 딕셔너리입니다.
# 각 키는 모델의 이름(예: "efficientvit-seg-b0")이며,
# 값은 다음 정보를 담은 튜플입니다:
# 1. 모델 생성 함수 (예: `efficientvit_seg_b0`)
# 2. 정규화(Normalization) 레이어의 epsilon 값
# 3. 기본으로 제공되는 사전 학습된 가중치 파일의 경로
REGISTERED_EFFICIENTVIT_SEG_MODEL: dict[str, tuple[Callable, float, str]] = {
    "efficientvit-seg-b0": (
        efficientvit_seg_b0,
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b0_cityscapes.pt",
    ),
    "efficientvit-seg-b1": (
        efficientvit_seg_b1,
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b1_cityscapes.pt",
    ),
    "efficientvit-seg-b2": (
        efficientvit_seg_b2,
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b2_cityscapes.pt",
    ),
    "efficientvit-seg-b3": (
        efficientvit_seg_b3,
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b3_cityscapes.pt",
    ),
    ############################################################################
    "efficientvit-seg-l1": (
        efficientvit_seg_l1,
        1e-7,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_l1_cityscapes.pt",
    ),
    "efficientvit-seg-l2": (
        efficientvit_seg_l2,
        1e-7,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_cityscapes.pt",
    ),
}

REGISTERED_SEGFORMER_MODEL: dict[str, tuple[Callable, float, Optional[str]]] = {
    "segformer-b0": (segformer_b0, 1e-6, None), # Pretrained path is None for now
    "segformer-b1": (segformer_b1, 1e-6, None),
    "segformer-b2": (segformer_b2, 1e-6, None),
    "segformer-b3": (segformer_b3, 1e-6, None),
    "segformer-b4": (segformer_b4, 1e-6, None),
    "segformer-b5": (segformer_b5, 1e-6, None),
}


def create_efficientvit_seg_model(
    name: str,
    dataset: str,
    pretrained: bool = True,
    weight_url: Optional[str] = None,
    n_classes: Optional[int] = None,
    **kwargs,
) -> EfficientViTSeg:
    """
    이름으로 EfficientViT 세그멘테이션 모델을 생성하고, 선택적으로 사전 학습된 가중치를 로드합니다.

    Args:
        name (str): 생성할 모델의 이름 (예: "efficientvit-seg-b0").
                      `REGISTERED_EFFICIENTVIT_SEG_MODEL`에 등록된 키여야 합니다.
        dataset (str): 모델 헤드를 구성하는 데 사용될 데이터셋 이름 (예: "cityscapes").
        pretrained (bool, optional): `True`이면 사전 학습된 가중치를 로드합니다. Defaults to True.
        weight_url (Optional[str], optional): 사용할 가중치 파일의 경로.
                                              `None`이면 등록된 기본 경로를 사용합니다. Defaults to None.
        n_classes (Optional[int], optional): 출력 클래스의 수. `None`이면 데이터셋의 기본값을 따릅니다. Defaults to None.
        **kwargs: 모델 생성 함수에 전달될 추가 인자.

    Returns:
        EfficientViTSeg: 생성되고 가중치가 로드된 모델 객체.
    """
    if name not in REGISTERED_EFFICIENTVIT_SEG_MODEL:
        raise ValueError(f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_SEG_MODEL.keys())}")

    # 1. 등록된 정보에서 모델 생성 함수, norm_eps, 기본 가중치 경로를 가져옵니다.
    model_builder, norm_eps, default_pt_path = REGISTERED_EFFICIENTVIT_SEG_MODEL[name]

    # 2. 모델 생성 함수를 호출하여 모델 아키텍처를 빌드합니다.
    model = model_builder(dataset=dataset, n_classes=n_classes, **kwargs)

    # 3. 모델의 모든 정규화 레이어의 epsilon 값을 설정합니다.
    set_norm_eps(model, norm_eps)

    # 4. 사전 학습된 가중치를 로드합니다.
    if pretrained:
        weight_path = weight_url or default_pt_path
        if weight_path is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        
        state_dict = load_state_dict_from_file(weight_path)
        model.load_state_dict(state_dict)
        
    return model

def create_segformer_model(
    name: str,
    pretrained: bool = True,
    weight_url: Optional[str] = None,
    n_classes: int = 19,
    **kwargs,
) -> nn.Module:
    """
    이름으로 SegFormer 모델을 생성하고, 선택적으로 사전 학습된 가중치를 로드합니다.

    Args:
        name (str): 생성할 모델의 이름 (예: "segformer-b0").
                      `REGISTERED_SEGFORMER_MODEL`에 등록된 키여야 합니다.
        pretrained (bool, optional): `True`이면 사전 학습된 가중치를 로드합니다. Defaults to True.
        weight_url (Optional[str], optional): 사용할 가중치 파일의 경로.
                                              `None`이면 등록된 기본 경로를 사용합니다. Defaults to None.
        n_classes (int, optional): 출력 클래스의 수. Defaults to 19.
        **kwargs: 모델 생성 함수에 전달될 추가 인자.

    Returns:
        nn.Module: 생성되고 가중치가 로드된 모델 객체.
    """
    if name not in REGISTERED_SEGFORMER_MODEL:
        raise ValueError(f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_SEGFORMER_MODEL.keys())}")

    model_builder, norm_eps, default_pt_path = REGISTERED_SEGFORMER_MODEL[name]

    model = model_builder(num_classes=n_classes, **kwargs)

    set_norm_eps(model, norm_eps)

    if pretrained:
        weight_path = weight_url or default_pt_path
        if weight_path is None:
            # 사전 학습된 가중치가 없어도 에러를 발생시키지 않고 경고만 출력
            print(f"Warning: No pretrained weights available for {name} in the model zoo. The model is initialized randomly.")
        else:
            state_dict = load_state_dict_from_file(weight_path)
            model.load_state_dict(state_dict)
            
    return model

