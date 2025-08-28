"""
이 파일은 Semantic Segmentation 태스크를 위한 데이터 처리 파이프라인을 정의합니다.
- 다양한 데이터셋(Cityscapes, ADE20K, Mapillary 등)을 처리하는 `Dataset` 클래스
- 데이터 증강(augmentation) 및 변환(transform)을 위한 여러 클래스
- 최종적으로 데이터로더(DataLoader)를 생성하는 `SegDataProvider` 클래스
등이 포함되어 있습니다.
"""
from typing import Any, Optional, Dict, List
import os
import json
from PIL import Image
import numpy as np
import random
import pathlib
import cv2
import math

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

from efficientvit.apps.data_provider import DataProvider

__all__ = ["SegDataProvider"] # create_dataset, create_data_loader 제거

class RemapClasses:
    """
    레이블 마스크의 클래스 ID를 새로운 ID로 리매핑하는 변환 클래스.
    JSON 형식의 매핑 파일을 사용하여 특정 클래스를 합치거나 인덱스를 변경할 때 사용됩니다.
    """
    def __init__(self, mapping_dict: Dict[int, int], ignore_index: int = 255):
        """
        Args:
            mapping_dict (Dict[int, int]): {old_id: new_id} 형식의 매핑 딕셔너리.
            ignore_index (int, optional): 매핑되지 않는 클래스에 할당될 무시 인덱스. Defaults to 255.
        """
        self.mapping_dict = mapping_dict
        self.ignore_index = ignore_index
        
        if not self.mapping_dict:
            self.remap_lut = None
            return
            
        max_old_id = max(self.mapping_dict.keys())
        self.remap_lut = np.full(max_old_id + 1, self.ignore_index, dtype=np.int64)
        for old_val, new_val in self.mapping_dict.items():
            self.remap_lut[old_val] = new_val

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.remap_lut is None:
            return sample
            
        label_img = sample['label']
        label_array = np.array(label_img, dtype=np.int64)
        
        max_lut_idx = len(self.remap_lut) - 1
        label_array[label_array > max_lut_idx] = self.ignore_index
        
        new_label_array = self.remap_lut[label_array]
        
        sample['label'] = Image.fromarray(new_label_array.astype(np.uint8), mode=label_img.mode)
        return sample

class SegCompose:
    """여러 변환(transform)들을 순차적으로 적용하기 위한 Compose 클래스."""
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

class SegToTensor:
    """PIL Image를 PyTorch 텐서로 변환합니다."""
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = F.to_tensor(sample['image'])
        label = torch.from_numpy(np.array(sample['label'])).long()
        return {'image': image, 'label': label}

class SegNormalize:
    """이미지 텐서를 정규화합니다."""
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample

class SegRandomHorizontalFlip:
    """이미지와 레이블에 대해 랜덤 수평 뒤집기를 적용합니다."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            sample['image'] = F.hflip(sample['image'])
            sample['label'] = F.hflip(sample['label'])
        return sample

class SegRandomResizedCrop:
    """이미지와 레이블에 대해 랜덤 리사이즈 및 크롭을 적용합니다."""
    def __init__(self, size: int, scale: tuple[float, float] = (0.5, 2.0)):
        self.size = (size, size)
        self.scale = scale

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        i, j, h, w = transforms.RandomResizedCrop.get_params(sample['image'], self.scale, ratio=(0.75, 1.33))
        sample['image'] = F.resized_crop(sample['image'], i, j, h, w, self.size, transforms.InterpolationMode.BILINEAR)
        sample['label'] = F.resized_crop(sample['label'], i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)
        return sample

class SegResize:
    """이미지와 레이블을 주어진 크기로 리사이즈합니다."""
    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['image'] = F.resize(sample['image'], self.size, transforms.InterpolationMode.BILINEAR)
        sample['label'] = F.resize(sample['label'], self.size, transforms.InterpolationMode.NEAREST)
        return sample

class SegCenterCrop:
    """이미지와 레이블의 중앙을 크롭합니다."""
    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['image'] = F.center_crop(sample['image'], self.size)
        sample['label'] = F.center_crop(sample['label'], self.size)
        return sample

class SegmentationDataset(Dataset):
    """
    일반적인 세그멘테이션 데이터셋을 위한 PyTorch Dataset 클래스.
    'images'와 'labels' 폴더 구조를 가진 데이터셋에 사용될 수 있습니다.
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
        classes: Optional[list] = None,
        class_colors: Optional[list] = None,
    ):
        """
        Args:
            root (str): 데이터셋의 루트 디렉토리.
            split (str): 'train', 'val' 등 데이터셋의 분할.
            transform (Optional[callable], optional): 샘플에 적용할 변환. Defaults to None.
            image_dir_name (str, optional): 이미지 폴더 이름. Defaults to "images".
            label_dir_name (str, optional): 레이블 폴더 이름. Defaults to "labels".
            image_suffix (str, optional): 이미지 파일 확장자. Defaults to ".jpg".
            label_suffix (str, optional): 레이블 파일 확장자. Defaults to ".png".
            classes (Optional[list], optional): 클래스 이름 리스트. Defaults to None.
            class_colors (Optional[list], optional): 클래스 색상 리스트. Defaults to None.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = classes
        self.class_colors = class_colors

        image_dir = os.path.join(self.root, self.split, image_dir_name)
        label_dir = os.path.join(self.root, self.split, label_dir_name)

        for img_name in sorted(os.listdir(image_dir)):
            if not img_name.endswith(image_suffix):
                continue
            self.images.append(os.path.join(image_dir, img_name))
            label_name = img_name.replace(image_suffix, label_suffix)
            self.labels.append(os.path.join(label_dir, label_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample["index"] = idx
        sample["image_path"] = img_path
        return sample

class SegDataProvider(DataProvider):
    """
    세그멘테이션 태스크를 위한 데이터 프로바이더.

    `DataProvider`를 상속받아, 학습/검증용 데이터셋과 데이터로더를 구성하고
    데이터 증강 및 변환 파이프라인을 구축하는 역할을 총괄합니다.
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
        n_classes: int = 19,
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        image_suffix: str = ".jpg",
        label_suffix: str = ".png",
        train_split: str = "train",
        val_split: str = "val",
        class_mapping_path: Optional[str] = None,
        class_definitions_file: Optional[str] = None,
    ):
        """
        Args:
            data_dir (Optional[str], optional): 데이터셋 루트 디렉토리.
            train_batch_size (int, optional): 학습용 배치 사이즈.
            test_batch_size (int, optional): 테스트용 배치 사이즈.
            valid_size (Optional[int | float], optional): 검증셋 크기 또는 비율.
            n_worker (int, optional): 데이터 로딩에 사용할 워커 프로세스 수.
            image_size (int | list[int], optional): 이미지 크기.
            num_replicas (Optional[int], optional): 분산 학습 시 전체 GPU 수.
            rank (Optional[int], optional): 분산 학습 시 현재 GPU의 순위.
            train_ratio (Optional[float], optional): 전체 데이터 중 학습에 사용할 비율.
            drop_last (bool, optional): 마지막 배치가 배치 사이즈보다 작을 경우 버릴지 여부.
            n_classes (int, optional): 클래스 개수.
            image_dir_name (str, optional): 이미지 폴더 이름.
            label_dir_name (str, optional): 레이블 폴더 이름.
            image_suffix (str, optional): 이미지 파일 확장자.
            label_suffix (str, optional): 레이블 파일 확장자.
            train_split (str, optional): 학습용 데이터 분할 이름.
            val_split (str, optional): 검증용 데이터 분할 이름.
            class_mapping_path (Optional[str], optional): 클래스 리매핑 JSON 파일 경로.
            class_definitions_file (Optional[str], optional): 커스텀 클래스 정의 JSON 파일 경로.
        """
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.image_dir_name = image_dir_name
        self.label_dir_name = label_dir_name
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.train_split = train_split
        self.val_split = val_split

        self.class_mapping = None
        if class_mapping_path:
            with open(class_mapping_path, 'r') as f:
                self.class_mapping = {int(k): v for k, v in json.load(f).items()}

        if class_definitions_file:
            with open(class_definitions_file, 'r', encoding='utf-8') as f:
                defs = json.load(f)
                self.classes = tuple(defs['classes'])
                self.class_colors = tuple(map(tuple, defs['class_colors']))
        else:
            self.classes = [f"class_{i}" for i in range(self.n_classes)]
            self.class_colors = None # Or generate random colors

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
        """학습용 데이터 변환 파이프라인을 구축합니다."""
        image_size = self.image_size if image_size is None else image_size
        
        train_transforms = []
        if self.class_mapping:
            train_transforms.append(RemapClasses(self.class_mapping))

        train_transforms.extend([
            SegRandomResizedCrop(image_size[0], scale=(0.5, 2.0)),
            SegRandomHorizontalFlip(),
            SegToTensor(),
            SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return SegCompose(train_transforms)

    def build_valid_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        """검증/테스트용 데이터 변환 파이프라인을 구축합니다."""
        image_size = (self.active_image_size if image_size is None else image_size)[0]
        
        valid_transforms = []
        if self.class_mapping:
            valid_transforms.append(RemapClasses(self.class_mapping))

        valid_transforms.extend([
            SegResize(image_size),
            SegCenterCrop(image_size),
            SegToTensor(),
            SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return SegCompose(valid_transforms)

    def build_datasets(self) -> tuple[Dataset, Dataset, None]:
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
            classes=self.classes,
            class_colors=self.class_colors,
        )
        val_dataset = SegmentationDataset(
            root=self.data_dir,
            split=self.val_split,
            transform=valid_transform,
            image_dir_name=self.image_dir_name,
            label_dir_name=self.label_dir_name,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            classes=self.classes,
            class_colors=self.class_colors,
        )
        
        return train_dataset, val_dataset, None