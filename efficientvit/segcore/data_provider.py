from typing import Any, Optional
import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from efficientvit.apps.data_provider import DataProvider

__all__ = ["SegDataProvider"]

class SegmentationDataset(Dataset):
    """
    시맨틱 세그멘테이션 작업을 위한 일반 데이터셋 클래스.
    제공된 구조에 따라 루트 디렉토리에서 이미지와 레이블을 로드합니다.
    """
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[callable] = None,
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        image_suffix: str = ".jpg",
        label_suffix: str = ".png",
    ):
        """
        데이터셋을 초기화합니다.

        Args:
            root (str): 데이터셋 루트 디렉토리 경로.
            split (str): 로드할 데이터 분할 (예: 'train', 'validation').
            transform (callable, optional): 이미지와 레이블에 적용할 전처리 함수. Defaults to None.
            image_dir_name (str): 이미지를 포함하는 디렉토리 이름.
            label_dir_name (str): 레이블을 포함하는 디렉토리 이름.
            image_suffix (str): 이미지 파일의 접미사.
            label_suffix (str): 레이블 파일의 접미사.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # 이미지 및 레이블 경로 설정
        image_dir = os.path.join(self.root, self.split, image_dir_name)
        label_dir = os.path.join(self.root, self.split, label_dir_name)

        # 이미지 및 레이블 파일 경로 목록 생성
        for img_name in sorted(os.listdir(image_dir)):
            if not img_name.endswith(image_suffix):
                continue
            self.images.append(os.path.join(image_dir, img_name))
            label_name = img_name.replace(image_suffix, label_suffix)
            self.labels.append(os.path.join(label_dir, label_name))

    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        지정된 인덱스에 해당하는 샘플(이미지, 레이블)을 로드합니다.

        Args:
            idx (int): 샘플의 인덱스.

        Returns:
            dict: 'image'와 'label'을 포함하는 딕셔너리.
        """
        img_path = self.images[idx]
        lbl_path = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class SegDataProvider(DataProvider):
    """
    일반 세그멘테이션 데이터셋을 위한 데이터 프로바이더.
    설정에 따라 데이터 로더와 증강을 설정합니다.
    """

    name = "seg"
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_batch_size=16,
        test_batch_size=16,
        valid_size: Optional[int | float] = None,
        n_worker=8,
        image_size: int | list[int] = 512,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
        # 일반 데이터셋 처리를 위한 새로운 파라미터
        n_classes: int = 19,  # 기본값은 Cityscapes, 설정 파일에 의해 덮어쓰여야 함
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        image_suffix: str = ".jpg",
        label_suffix: str = ".png",
        train_split: str = "train",
        val_split: str = "val",
    ):
        # 데이터셋별 속성 저장
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.image_dir_name = image_dir_name
        self.label_dir_name = label_dir_name
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.train_split = train_split
        self.val_split = val_split

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )

    def build_train_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        """학습 데이터에 적용할 이미지 변환 파이프라인을 구축합니다."""
        image_size = self.image_size if image_size is None else image_size
        
        train_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(train_transforms)

    def build_valid_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        """검증 데이터에 적용할 이미지 변환 파이프라인을 구축합니다."""
        image_size = (self.active_image_size if image_size is None else image_size)[0]
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_datasets(self) -> tuple[Any, Any, Any]:
        """학습 및 검증 데이터셋 객체를 생성합니다."""
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()

        train_dataset = SegmentationDataset(
            root=self.data_dir,
            split=self.train_split,
            transform=train_transform,
            image_dir_name=self.image_dir_name,
            label_dir_name=self.label_dir_name,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
        )
        val_dataset = SegmentationDataset(
            root=self.data_dir,
            split=self.val_split,
            transform=valid_transform,
            image_dir_name=self.image_dir_name,
            label_dir_name=self.label_dir_name,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
        )
        
        # 테스트 데이터셋은 사용하지 않으므로 None으로 설정합니다.
        return train_dataset, val_dataset, None