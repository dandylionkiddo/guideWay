import copy

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from efficientvit.models.utils import torch_random_choices

__all__ = [
    "RRSController",
    "get_interpolate",
    "MyRandomResizedCrop",
]


class RRSController:
    ACTIVE_SIZE = (224, 224)
    IMAGE_SIZE_LIST = [(224, 224)]

    CHOICE_LIST = []

    @staticmethod
    def get_candidates() -> list[tuple[int, int]]:
        return copy.deepcopy(RRSController.IMAGE_SIZE_LIST)

    @staticmethod
    def sample_resolution(batch_id: int | None) -> None:
        """
        배치 진입 시 활성 해상도 선택.
        - CHOICE_LIST가 비어있으면(워커 초기화 전/전달 실패) IMAGE_SIZE_LIST를 폴백으로 사용
        - batch_id가 범위를 넘으면 안전하게 modulo
        """
        # 사용 가능한 소스 결정
        choices = RRSController.CHOICE_LIST if RRSController.CHOICE_LIST else RRSController.IMAGE_SIZE_LIST
        if not choices:
            raise RuntimeError("RRSController: IMAGE_SIZE_LIST/CHOICE_LIST가 비어 있습니다.")

        if batch_id is None:
            RRSController.ACTIVE_SIZE = choices[-1]
        else:
            RRSController.ACTIVE_SIZE = choices[batch_id % len(choices)]

    @staticmethod
    def set_epoch(epoch: int, batch_per_epoch: int) -> None:
        # 후보 안전성 보장
        cands = RRSController.get_candidates()
        if not cands:
            # 최소 한 개는 있어야 함
            cands = [RRSController.ACTIVE_SIZE]
            RRSController.IMAGE_SIZE_LIST = cands

        # 배치 수가 0이거나 음수인 경우 방어
        if not isinstance(batch_per_epoch, int) or batch_per_epoch <= 0:
            RRSController.CHOICE_LIST = []
            RRSController.ACTIVE_SIZE = cands[-1]
            return

        # 에폭 고정 시드로 배치 수 만큼 시퀀스 생성
        g = torch.Generator()
        g.manual_seed(epoch)
        RRSController.CHOICE_LIST = torch_random_choices(cands, g, batch_per_epoch)
        # 첫 배치 기본값 설정
        RRSController.ACTIVE_SIZE = RRSController.CHOICE_LIST[0] if RRSController.CHOICE_LIST else cands[-1]


def get_interpolate(name: str) -> F.InterpolationMode:
    mapping = {
        "nearest": F.InterpolationMode.NEAREST,
        "bilinear": F.InterpolationMode.BILINEAR,
        "bicubic": F.InterpolationMode.BICUBIC,
        "box": F.InterpolationMode.BOX,
        "hamming": F.InterpolationMode.HAMMING,
        "lanczos": F.InterpolationMode.LANCZOS,
    }
    if name in mapping:
        return mapping[name]
    elif name == "random":
        return torch_random_choices(
            [
                F.InterpolationMode.NEAREST,
                F.InterpolationMode.BILINEAR,
                F.InterpolationMode.BICUBIC,
                F.InterpolationMode.BOX,
                F.InterpolationMode.HAMMING,
                F.InterpolationMode.LANCZOS,
            ],
        )
    else:
        raise NotImplementedError


class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "random",
    ):
        super(MyRandomResizedCrop, self).__init__(224, scale, ratio)
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        i, j, h, w = self.get_params(img, list(self.scale), list(self.ratio))
        target_size = RRSController.ACTIVE_SIZE
        return F.resized_crop(img, i, j, h, w, list(target_size), get_interpolate(self.interpolation))

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += f"(\n\tsize={RRSController.get_candidates()},\n"
        format_string += f"\tscale={tuple(round(s, 4) for s in self.scale)},\n"
        format_string += f"\tratio={tuple(round(r, 4) for r in self.ratio)},\n"
        format_string += f"\tinterpolation={self.interpolation})"
        return format_string
