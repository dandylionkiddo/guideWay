from typing import Any, Optional
import os
import json
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from efficientvit.apps.data_provider import DataProvider

__all__ = ["MapillaryDataProvider"]

class MapillaryDataset(Dataset):
    """
    Mapillary Vistas 데이터셋을 로드하기 위한 PyTorch Dataset 클래스.
    데이터셋 루트 디렉토리 구조에 맞게 이미지와 레이블(마스크) 경로를 관리합니다.
    """
    def __init__(self, root, split='train', transform=None):
        """
        데이터셋을 초기화합니다.

        Args:
            root (str): 데이터셋의 루트 디렉토리 경로.
            split (str, optional): 로드할 데이터 분할 ('training' 또는 'validation'). Defaults to 'train'.
            transform (callable, optional): 이미지와 레이블에 적용할 전처리 함수. Defaults to None.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # config.json에서 레이블 정보를 로드합니다.
        with open(os.path.join(self.root, 'config.json')) as f:
            config = json.load(f)
        self.labels_info = config['labels']

        # 이미지와 레이블 디렉토리 경로를 설정합니다.
        image_dir = os.path.join(self.root, split, 'images')
        label_dir = os.path.join(self.root, split, 'labels')

        # 이미지와 레이블 파일 경로 리스트를 생성합니다.
        for img_name in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_name))
            self.labels.append(os.path.join(label_dir, img_name.replace('.jpg', '.png')))

    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        지정된 인덱스(idx)에 해당하는 샘플(이미지, 레이블)을 로드합니다.

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

class MapillaryDataProvider(DataProvider):
    """
    Mapillary 데이터셋을 위한 데이터 프로바이더.
    학습 및 검증 데이터로더와 데이터 증강(augmentation)을 설정합니다.
    """
    name = "mapillary"
    
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
    ):
        self.data_dir = data_dir
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
        """학습 데이터에 적용할 이미지 변환(augmentation) 파이프라인을 구축합니다."""
        image_size = self.image_size if image_size is None else image_size
        
        train_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 2.0)), # 이미지 크기를 랜덤하게 조절하고 자릅니다.
            transforms.RandomHorizontalFlip(), # 50% 확률로 이미지를 좌우 반전시킵니다.
            transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환합니다.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 통계로 정규화합니다.
        ]
        return transforms.Compose(train_transforms)

    def build_valid_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        """검증 데이터에 적용할 이미지 변환 파이프라인을 구축합니다."""
        image_size = (self.active_image_size if image_size is None else image_size)[0]
        return transforms.Compose(
            [
                transforms.Resize(image_size), # 이미지 크기를 조절합니다.
                transforms.CenterCrop(image_size), # 중앙을 기준으로 이미지를 자릅니다.
                transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환합니다.
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 통계로 정규화합니다.
            ]
        )

    def build_datasets(self) -> tuple[Any, Any, Any]:
        """학습 및 검증 데이터셋 객체를 생성합니다."""
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()

        train_dataset = MapillaryDataset(root=self.data_dir, split='training', transform=train_transform)
        val_dataset = MapillaryDataset(root=self.data_dir, split='validation', transform=valid_transform)
        
        # 테스트 데이터셋은 사용하지 않으므로 None으로 설정합니다.
        return train_dataset, val_dataset, None