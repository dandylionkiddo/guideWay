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

__all__ = ["SegDataProvider", "create_dataset", "create_data_loader"]


# region Evaluation-specific Classes (from old eval script)
class EvalResize:
    def __init__(
        self,
        crop_size: Optional[tuple[int, int]],
        interpolation: Optional[int] = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.crop_size is None or self.interpolation is None:
            return feed_dict

        image, target = feed_dict["data"], feed_dict["label"]
        height, width = self.crop_size

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
            target = cv2.resize(
                target,
                dsize=(width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return {
            "data": image,
            "label": target,
        }


class EvalToTensor:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        image, mask = feed_dict["data"], feed_dict["label"]
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {
            "data": image,
            "label": mask,
        }


class CityscapesDataset(Dataset):
    classes = (
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
    )
    class_colors = (
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
    )
    label_map = np.array(
        (-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18)
    )

    def __init__(self, data_dir: str, crop_size: Optional[tuple[int, int]] = None):
        super().__init__()
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                if pathlib.Path(fname).suffix not in [".png"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace(
                    "_leftImg8bit.", "_gtFine_labelIds."
                )
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples
        self.transform = transforms.Compose(
            [
                EvalResize(crop_size),
                EvalToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.label_map[mask]
        feed_dict = {"data": image, "label": mask}
        feed_dict = self.transform(feed_dict)
        return {"index": index, "image_path": image_path, **feed_dict}


class ADE20KDataset(Dataset):
    classes = ("wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag")
    class_colors = ([120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255])

    def __init__(self, data_dir: str, crop_size=512):
        super().__init__()
        self.crop_size = crop_size
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                if pathlib.Path(fname).suffix not in [".jpg"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/images/", "/annotations/")
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples
        self.transform = transforms.Compose(
            [EvalToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.int64) - 1
        h, w = image.shape[:2]
        if h < w:
            th = self.crop_size
            tw = math.ceil(w / h * th / 32) * 32
        else:
            tw = self.crop_size
            th = math.ceil(h / w * tw / 32) * 32
        if th != h or tw != w:
            image = cv2.resize(image, dsize=(tw, th), interpolation=cv2.INTER_CUBIC)
        feed_dict = {"data": image, "label": mask}
        feed_dict = self.transform(feed_dict)
        return {"index": index, "image_path": image_path, **feed_dict}


class MapillaryDataset(Dataset):
    # Default classes (124)
    classes = ("Bird", "Ground Animal", "Ambiguous Barrier", "Concrete Block", "Curb", "Fence", "Guard Rail", "Barrier", "Road Median", "Road Side", "Lane Separator", "Temporary Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Driveway", "Parking", "Parking Aisle", "Pedestrian Area", "Rail Track", "Road", "Road Shoulder", "Service Lane", "Sidewalk", "Traffic Island", "Bridge", "Building", "Garage", "Tunnel", "Person", "Person Group", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Dashed Line", "Lane Marking - Straight Line", "Lane Marking - Zigzag Line", "Lane Marking - Ambiguous", "Lane Marking - Arrow (Left)", "Lane Marking - Arrow (Other)", "Lane Marking - Arrow (Right)", "Lane Marking - Arrow (Split Left or Straight)", "Lane Marking - Arrow (Split Right or Straight)", "Lane Marking - Arrow (Straight)", "Lane Marking - Crosswalk", "Lane Marking - Give Way (Row)", "Lane Marking - Give Way (Single)", "Lane Marking - Hatched (Chevron)", "Lane Marking - Hatched (Diagonal)", "Lane Marking - Other", "Lane Marking - Stop Line", "Lane Marking - Symbol (Bicycle)", "Lane Marking - Symbol (Other)", "Lane Marking - Text", "Lane Marking (only) - Dashed Line", "Lane Marking (only) - Crosswalk", "Lane Marking (only) - Other", "Lane Marking (only) - Test", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Mailbox", "Manhole", "Parking Meter", "Phone Booth", "Pothole", "Signage - Advertisement", "Signage - Ambiguous", "Signage - Back", "Signage - Information", "Signage - Other", "Signage - Store", "Street Light", "Pole", "Pole Group", "Traffic Sign Frame", "Utility Pole", "Traffic Cone", "Traffic Light - General (Single)", "Traffic Light - Pedestrians", "Traffic Light - General (Upright)", "Traffic Light - General (Horizontal)", "Traffic Light - Cyclists", "Traffic Light - Other", "Traffic Sign - Ambiguous", "Traffic Sign (Back)", "Traffic Sign - Direction (Back)", "Traffic Sign - Direction (Front)", "Traffic Sign (Front)", "Traffic Sign - Parking", "Traffic Sign - Temporary (Back)", "Traffic Sign - Temporary (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Vehicle Group", "Wheeled Slow", "Water Valve", "Car Mount", "Dynamic", "Ego Vehicle", "Ground", "Static", "Unlabeled")
    class_colors = ([165, 42, 42], [0, 192, 0], [250, 170, 31], [250, 170, 32], [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150], [250, 170, 33], [250, 170, 34], [128, 128, 128], [250, 170, 35], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 36], [250, 170, 160], [250, 170, 37], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [110, 110, 110], [244, 35, 232], [128, 196, 128], [150, 100, 100], [70, 70, 70], [150, 150, 150], [150, 120, 90], [220, 20, 60], [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200], [255, 255, 255], [255, 255, 255], [250, 170, 29], [250, 170, 28], [250, 170, 26], [250, 170, 25], [250, 170, 24], [250, 170, 22], [250, 170, 21], [250, 170, 20], [255, 255, 255], [250, 170, 19], [250, 170, 18], [250, 170, 12], [250, 170, 11], [255, 255, 255], [255, 255, 255], [250, 170, 16], [250, 170, 15], [250, 170, 15], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [100, 128, 160], [20, 20, 255], [142, 0, 0], [70, 100, 150], [250, 171, 30], [250, 172, 30], [250, 173, 30], [250, 174, 30], [250, 175, 30], [250, 176, 30], [210, 170, 100], [153, 153, 153], [153, 153, 153], [128, 128, 128], [0, 0, 80], [210, 60, 60], [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30], [192, 192, 192], [192, 192, 192], [192, 192, 192], [220, 220, 0], [220, 220, 0], [0, 0, 196], [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 142], [0, 0, 192], [170, 170, 170], [32, 32, 32], [111, 74, 0], [120, 10, 10], [81, 0, 81], [111, 111, 0], [0, 0, 0])

    def __init__(self, 
                 data_dir: str, 
                 crop_size: Optional[tuple[int, int]] = None, 
                 class_mapping_file: Optional[str] = None, 
                 class_definitions_file: Optional[str] = None):
        super().__init__()
        self.ignore_index = -1  # Default ignore index

        # Load class definitions and mapping files if provided
        self.remap_lut = None
        if class_definitions_file:
            with open(class_definitions_file, 'r', encoding='utf-8') as f:
                defs = json.load(f)
                self.classes = tuple(defs['classes'])
                self.class_colors = tuple(map(tuple, defs['class_colors']))

        if class_mapping_file:
            self.ignore_index = 255  # Remapped ignore index
            with open(class_mapping_file, 'r') as f:
                mapping_dict = {int(k): v for k, v in json.load(f).items()}
            # Create a lookup table for fast remapping
            max_id = max(mapping_dict.keys())
            self.remap_lut = np.arange(max_id + 1, dtype=np.int64)
            self.remap_lut.fill(255)
            for old_val, new_val in mapping_dict.items():
                self.remap_lut[old_val] = new_val

        # load samples
        samples = []
        image_dir = os.path.join(data_dir, "images")
        label_dir = os.path.join(data_dir, "labels")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(label_dir):
            for fname in sorted(os.listdir(image_dir)):
                if fname.lower().endswith('.jpg'):
                    samples.append((os.path.join(image_dir, fname), None))
        else:
            for fname in sorted(os.listdir(image_dir)):
                if fname.lower().endswith('.jpg'):
                    image_path = os.path.join(image_dir, fname)
                    label_fname = os.path.splitext(fname)[0] + '.png'
                    label_path = os.path.join(label_dir, label_fname)
                    if os.path.exists(label_path):
                        samples.append((image_path, label_path))
        self.samples = samples
        self.transform = transforms.Compose(
            [
                EvalResize(crop_size),
                EvalToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        if mask_path is not None:
            mask = np.array(Image.open(mask_path), dtype=np.int64)
            # Apply remapping if the lookup table is available
            if self.remap_lut is not None:
                # Ensure mask values are within the LUT's bounds
                max_lut_idx = len(self.remap_lut) - 1
                mask[mask > max_lut_idx] = max_lut_idx # or some ignore value
                mask = self.remap_lut[mask]
            else:
                mask[mask >= len(self.classes)] = -1 # Default behavior
        else:
            mask = np.full(image.shape[:2], -1, dtype=np.int64)
        feed_dict = {"data": image, "label": mask}
        feed_dict = self.transform(feed_dict)
        return {"index": index, "image_path": image_path, **feed_dict}

# endregion

# region Factory Functions for eval
def create_dataset(config: dict) -> Dataset:
    dataset_config = config["dataset"]
    name = dataset_config["name"]
    path = os.path.expanduser(dataset_config["path"])
    crop_size = dataset_config.get("crop_size")

    if name == "cityscapes":
        eval_crop_size = (crop_size, crop_size * 2) if crop_size else None
        return CityscapesDataset(path, crop_size=eval_crop_size)
    elif name == "ade20k":
        return ADE20KDataset(path, crop_size=crop_size)
    elif name == "mapillary":
        eval_crop_size = (crop_size, crop_size) if crop_size else None
        mapping_file = dataset_config.get("class_mapping_file")
        definitions_file = dataset_config.get("class_definitions_file")
        return MapillaryDataset(
            path, 
            crop_size=eval_crop_size, 
            class_mapping_file=mapping_file,
            class_definitions_file=definitions_file
        )
    else:
        return SegmentationDataset(root=path, split=config.get("split", "val"))

def create_data_loader(dataset: Dataset, config: dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["runtime"]["workers"],
        pin_memory=True,
        drop_last=False,
    )
# endregion


class RemapClasses:
    def __init__(self, mapping_dict: Dict[int, int], ignore_index: int = 255):
        self.mapping_dict = mapping_dict
        self.ignore_index = ignore_index
        
        # Create a lookup table for fast remapping
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
        
        # Ensure mask values are within the LUT's bounds, otherwise map to ignore_index
        max_lut_idx = len(self.remap_lut) - 1
        # Pixels with values greater than the max known ID are mapped to ignore_index
        label_array[label_array > max_lut_idx] = self.ignore_index
        
        # Apply the remapping using the lookup table
        new_label_array = self.remap_lut[label_array]
        
        sample['label'] = Image.fromarray(new_label_array.astype(np.uint8), mode=label_img.mode)
        return sample

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
        class_mapping_path: Optional[str] = None,
    ):
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
