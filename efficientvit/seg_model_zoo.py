from typing import Callable, Optional

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

__all__ = ["create_efficientvit_seg_model"]


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


def create_efficientvit_seg_model(
    name: str,
    dataset: str,
    pretrained=True,
    weight_url: Optional[str] = None,
    n_classes: Optional[int] = None,
    **kwargs,
) -> EfficientViTSeg:
    if name not in REGISTERED_EFFICIENTVIT_SEG_MODEL:
        raise ValueError(f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_SEG_MODEL.keys())}")
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SEG_MODEL[name]
        model = model_cls(dataset=dataset, n_classes=n_classes, **kwargs)
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url

    if pretrained:
        if weight_url is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model

