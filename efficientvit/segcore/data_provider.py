from typing import Any, Optional, Dict, List
import os
from PIL import Image
import numpy as np
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from efficientvit.apps.data_provider import DataProvider

__all__ = ["SegDataProvider"]

# 세그멘테이션을 위한 커스텀 변환 클래스들
class SegCompose:
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

class SegToTensor:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = F.to_tensor(sample['image'])
        label = torch.from_numpy(np.array(sample['label'])).long()
        return {'image': image, 'label': label}

class SegNormalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample

class SegRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            sample['image'] = F.hflip(sample['image'])
            sample['label'] = F.hflip(sample['label'])
        return sample

class SegRandomResizedCrop:
    def __init__(self, size: int, scale: tuple[float, float] = (0.5, 2.0)):
        self.size = (size, size)
        self.scale = scale

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        i, j, h, w = transforms.RandomResizedCrop.get_params(sample['image'], self.scale, ratio=(0.75, 1.33))
        sample['image'] = F.resized_crop(sample['image'], i, j, h, w, self.size, Image.BILINEAR)
        sample['label'] = F.resized_crop(sample['label'], i, j, h, w, self.size, Image.NEAREST)
        return sample

class SegResize:
    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['image'] = F.resize(sample['image'], self.size, Image.BILINEAR)
        sample['label'] = F.resize(sample['label'], self.size, Image.NEAREST)
        return sample

class SegCenterCrop:
    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['image'] = F.center_crop(sample['image'], self.size)
        sample['label'] = F.center_crop(sample['label'], self.size)
        return sample

class SegmentationDataset(Dataset):
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
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

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

        return sample

class SegDataProvider(DataProvider):
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
    ):
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
        image_size = self.image_size if image_size is None else image_size
        
        train_transforms = [
            SegRandomResizedCrop(image_size[0], scale=(0.5, 2.0)),
            SegRandomHorizontalFlip(),
            SegToTensor(),
            SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return SegCompose(train_transforms)

    def build_valid_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        image_size = (self.active_image_size if image_size is None else image_size)[0]
        return SegCompose(
            [
                SegResize(image_size),
                SegCenterCrop(image_size),
                SegToTensor(),
                SegNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_datasets(self) -> tuple[Any, Any, Any]:
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
        
        return train_dataset, val_dataset, None
