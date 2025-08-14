import argparse
import json
import math
import os
import pathlib
import sys
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import AverageMeter
from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


class Resize(object):
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


class ToTensor(object):
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


class SegIOU:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }


class CityscapesDataset(Dataset):
    classes = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
    class_colors = (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    )
    label_map = np.array(
        (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,  # road 7
            1,  # sidewalk 8
            -1,
            -1,
            2,  # building 11
            3,  # wall 12
            4,  # fence 13
            -1,
            -1,
            -1,
            5,  # pole 17
            -1,
            6,  # traffic light 19
            7,  # traffic sign 20
            8,  # vegetation 21
            9,  # terrain 22
            10,  # sky 23
            11,  # person 24
            12,  # rider 25
            13,  # car 26
            14,  # truck 27
            15,  # bus 28
            -1,
            -1,
            16,  # train 31
            17,  # motorcycle 32
            18,  # bicycle 33
        )
    )

    def __init__(self, data_dir: str, crop_size: Optional[tuple[int, int]] = None):
        super().__init__()

        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".png"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace(
                    "_leftImg8bit.", "_gtFine_labelIds."
                )
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        # build transform
        self.transform = transforms.Compose(
            [
                Resize(crop_size),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.label_map[mask]

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }


class ADE20KDataset(Dataset):
    classes = (
        "wall",
        "building",
        "sky",
        "floor",
        "tree",
        "ceiling",
        "road",
        "bed",
        "windowpane",
        "grass",
        "cabinet",
        "sidewalk",
        "person",
        "earth",
        "door",
        "table",
        "mountain",
        "plant",
        "curtain",
        "chair",
        "car",
        "water",
        "painting",
        "sofa",
        "shelf",
        "house",
        "sea",
        "mirror",
        "rug",
        "field",
        "armchair",
        "seat",
        "fence",
        "desk",
        "rock",
        "wardrobe",
        "lamp",
        "bathtub",
        "railing",
        "cushion",
        "base",
        "box",
        "column",
        "signboard",
        "chest of drawers",
        "counter",
        "sand",
        "sink",
        "skyscraper",
        "fireplace",
        "refrigerator",
        "grandstand",
        "path",
        "stairs",
        "runway",
        "case",
        "pool table",
        "pillow",
        "screen door",
        "stairway",
        "river",
        "bridge",
        "bookcase",
        "blind",
        "coffee table",
        "toilet",
        "flower",
        "book",
        "hill",
        "bench",
        "countertop",
        "stove",
        "palm",
        "kitchen island",
        "computer",
        "swivel chair",
        "boat",
        "bar",
        "arcade machine",
        "hovel",
        "bus",
        "towel",
        "light",
        "truck",
        "tower",
        "chandelier",
        "awning",
        "streetlight",
        "booth",
        "television receiver",
        "airplane",
        "dirt track",
        "apparel",
        "pole",
        "land",
        "bannister",
        "escalator",
        "ottoman",
        "bottle",
        "buffet",
        "poster",
        "stage",
        "van",
        "ship",
        "fountain",
        "conveyer belt",
        "canopy",
        "washer",
        "plaything",
        "swimming pool",
        "stool",
        "barrel",
        "basket",
        "waterfall",
        "tent",
        "bag",
        "minibike",
        "cradle",
        "oven",
        "ball",
        "food",
        "step",
        "tank",
        "trade name",
        "microwave",
        "pot",
        "animal",
        "bicycle",
        "lake",
        "dishwasher",
        "screen",
        "blanket",
        "sculpture",
        "hood",
        "sconce",
        "vase",
        "traffic light",
        "tray",
        "ashcan",
        "fan",
        "pier",
        "crt screen",
        "plate",
        "monitor",
        "bulletin board",
        "shower",
        "radiator",
        "glass",
        "clock",
        "flag",
    )
    class_colors = (
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    )

    def __init__(self, data_dir: str, crop_size=512):
        super().__init__()

        self.crop_size = crop_size
        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".jpg"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/images/", "/annotations/")
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        self.transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
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
            image = cv2.resize(
                image,
                dsize=(tw, th),
                interpolation=cv2.INTER_CUBIC,
            )

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }


class MapillaryDataset(Dataset):
    """Mapillary Vistas Dataset v2.0 - 124 classes (complete set)"""
    
    # 124개 모든 클래스 (config_v2.0.json 기반)
    classes = (
        "Bird",
        "Ground Animal",
        "Ambiguous Barrier",
        "Concrete Block",
        "Curb",
        "Fence",
        "Guard Rail",
        "Barrier",
        "Road Median",
        "Road Side",
        "Lane Separator",
        "Temporary Barrier",
        "Wall",
        "Bike Lane",
        "Crosswalk - Plain",
        "Curb Cut",
        "Driveway",
        "Parking",
        "Parking Aisle",
        "Pedestrian Area",
        "Rail Track",
        "Road",
        "Road Shoulder",
        "Service Lane",
        "Sidewalk",
        "Traffic Island",
        "Bridge",
        "Building",
        "Garage",
        "Tunnel",
        "Person",
        "Person Group",
        "Bicyclist",
        "Motorcyclist",
        "Other Rider",
        "Lane Marking - Dashed Line",
        "Lane Marking - Straight Line",
        "Lane Marking - Zigzag Line",
        "Lane Marking - Ambiguous",
        "Lane Marking - Arrow (Left)",
        "Lane Marking - Arrow (Other)",
        "Lane Marking - Arrow (Right)",
        "Lane Marking - Arrow (Split Left or Straight)",
        "Lane Marking - Arrow (Split Right or Straight)",
        "Lane Marking - Arrow (Straight)",
        "Lane Marking - Crosswalk",
        "Lane Marking - Give Way (Row)",
        "Lane Marking - Give Way (Single)",
        "Lane Marking - Hatched (Chevron)",
        "Lane Marking - Hatched (Diagonal)",
        "Lane Marking - Other",
        "Lane Marking - Stop Line",
        "Lane Marking - Symbol (Bicycle)",
        "Lane Marking - Symbol (Other)",
        "Lane Marking - Text",
        "Lane Marking (only) - Dashed Line",
        "Lane Marking (only) - Crosswalk",
        "Lane Marking (only) - Other",
        "Lane Marking (only) - Test",
        "Mountain",
        "Sand",
        "Sky",
        "Snow",
        "Terrain",
        "Vegetation",
        "Water",
        "Banner",
        "Bench",
        "Bike Rack",
        "Catch Basin",
        "CCTV Camera",
        "Fire Hydrant",
        "Junction Box",
        "Mailbox",
        "Manhole",
        "Parking Meter",
        "Phone Booth",
        "Pothole",
        "Signage - Advertisement",
        "Signage - Ambiguous",
        "Signage - Back",
        "Signage - Information",
        "Signage - Other",
        "Signage - Store",
        "Street Light",
        "Pole",
        "Pole Group",
        "Traffic Sign Frame",
        "Utility Pole",
        "Traffic Cone",
        "Traffic Light - General (Single)",
        "Traffic Light - Pedestrians",
        "Traffic Light - General (Upright)",
        "Traffic Light - General (Horizontal)",
        "Traffic Light - Cyclists",
        "Traffic Light - Other",
        "Traffic Sign - Ambiguous",
        "Traffic Sign (Back)",
        "Traffic Sign - Direction (Back)",
        "Traffic Sign - Direction (Front)",
        "Traffic Sign (Front)",
        "Traffic Sign - Parking",
        "Traffic Sign - Temporary (Back)",
        "Traffic Sign - Temporary (Front)",
        "Trash Can",
        "Bicycle",
        "Boat",
        "Bus",
        "Car",
        "Caravan",
        "Motorcycle",
        "On Rails",
        "Other Vehicle",
        "Trailer",
        "Truck",
        "Vehicle Group",
        "Wheeled Slow",
        "Water Valve",
        "Car Mount",
        "Dynamic",
        "Ego Vehicle",
        "Ground",
        "Static",
        "Unlabeled",
    )
    
    # 124개 클래스 색상 팔레트 (config_v2.0.json 기반)
    class_colors = (
        [165, 42, 42],      # Bird
        [0, 192, 0],        # Ground Animal
        [250, 170, 31],     # Ambiguous Barrier
        [250, 170, 32],     # Concrete Block
        [196, 196, 196],    # Curb
        [190, 153, 153],    # Fence
        [180, 165, 180],    # Guard Rail
        [90, 120, 150],     # Barrier
        [250, 170, 33],     # Road Median
        [250, 170, 34],     # Road Side
        [128, 128, 128],    # Lane Separator
        [250, 170, 35],     # Temporary Barrier
        [102, 102, 156],    # Wall
        [128, 64, 255],     # Bike Lane
        [140, 140, 200],    # Crosswalk - Plain
        [170, 170, 170],    # Curb Cut
        [250, 170, 36],     # Driveway
        [250, 170, 160],    # Parking
        [250, 170, 37],     # Parking Aisle
        [96, 96, 96],       # Pedestrian Area
        [230, 150, 140],    # Rail Track
        [128, 64, 128],     # Road
        [110, 110, 110],    # Road Shoulder
        [110, 110, 110],    # Service Lane
        [244, 35, 232],     # Sidewalk
        [128, 196, 128],    # Traffic Island
        [150, 100, 100],    # Bridge
        [70, 70, 70],       # Building
        [150, 150, 150],    # Garage
        [150, 120, 90],     # Tunnel
        [220, 20, 60],      # Person
        [220, 20, 60],      # Person Group
        [255, 0, 0],        # Bicyclist
        [255, 0, 100],      # Motorcyclist
        [255, 0, 200],      # Other Rider
        [255, 255, 255],    # Lane Marking - Dashed Line
        [255, 255, 255],    # Lane Marking - Straight Line
        [250, 170, 29],     # Lane Marking - Zigzag Line
        [250, 170, 28],     # Lane Marking - Ambiguous
        [250, 170, 26],     # Lane Marking - Arrow (Left)
        [250, 170, 25],     # Lane Marking - Arrow (Other)
        [250, 170, 24],     # Lane Marking - Arrow (Right)
        [250, 170, 22],     # Lane Marking - Arrow (Split Left or Straight)
        [250, 170, 21],     # Lane Marking - Arrow (Split Right or Straight)
        [250, 170, 20],     # Lane Marking - Arrow (Straight)
        [255, 255, 255],    # Lane Marking - Crosswalk
        [250, 170, 19],     # Lane Marking - Give Way (Row)
        [250, 170, 18],     # Lane Marking - Give Way (Single)
        [250, 170, 12],     # Lane Marking - Hatched (Chevron)
        [250, 170, 11],     # Lane Marking - Hatched (Diagonal)
        [255, 255, 255],    # Lane Marking - Other
        [255, 255, 255],    # Lane Marking - Stop Line
        [250, 170, 16],     # Lane Marking - Symbol (Bicycle)
        [250, 170, 15],     # Lane Marking - Symbol (Other)
        [250, 170, 15],     # Lane Marking - Text
        [255, 255, 255],    # Lane Marking (only) - Dashed Line
        [255, 255, 255],    # Lane Marking (only) - Crosswalk
        [255, 255, 255],    # Lane Marking (only) - Other
        [255, 255, 255],    # Lane Marking (only) - Test
        [64, 170, 64],      # Mountain
        [230, 160, 50],     # Sand
        [70, 130, 180],     # Sky
        [190, 255, 255],    # Snow
        [152, 251, 152],    # Terrain
        [107, 142, 35],     # Vegetation
        [0, 170, 30],       # Water
        [255, 255, 128],    # Banner
        [250, 0, 30],       # Bench
        [100, 140, 180],    # Bike Rack
        [220, 128, 128],    # Catch Basin
        [222, 40, 40],      # CCTV Camera
        [100, 170, 30],     # Fire Hydrant
        [40, 40, 40],       # Junction Box
        [33, 33, 33],       # Mailbox
        [100, 128, 160],    # Manhole
        [20, 20, 255],      # Parking Meter
        [142, 0, 0],        # Phone Booth
        [70, 100, 150],     # Pothole
        [250, 171, 30],     # Signage - Advertisement
        [250, 172, 30],     # Signage - Ambiguous
        [250, 173, 30],     # Signage - Back
        [250, 174, 30],     # Signage - Information
        [250, 175, 30],     # Signage - Other
        [250, 176, 30],     # Signage - Store
        [210, 170, 100],    # Street Light
        [153, 153, 153],    # Pole
        [153, 153, 153],    # Pole Group
        [128, 128, 128],    # Traffic Sign Frame
        [0, 0, 80],         # Utility Pole
        [210, 60, 60],      # Traffic Cone
        [250, 170, 30],     # Traffic Light - General (Single)
        [250, 170, 30],     # Traffic Light - Pedestrians
        [250, 170, 30],     # Traffic Light - General (Upright)
        [250, 170, 30],     # Traffic Light - General (Horizontal)
        [250, 170, 30],     # Traffic Light - Cyclists
        [250, 170, 30],     # Traffic Light - Other
        [192, 192, 192],    # Traffic Sign - Ambiguous
        [192, 192, 192],    # Traffic Sign (Back)
        [192, 192, 192],    # Traffic Sign - Direction (Back)
        [220, 220, 0],      # Traffic Sign - Direction (Front)
        [220, 220, 0],      # Traffic Sign (Front)
        [0, 0, 196],        # Traffic Sign - Parking
        [192, 192, 192],    # Traffic Sign - Temporary (Back)
        [220, 220, 0],      # Traffic Sign - Temporary (Front)
        [140, 140, 20],     # Trash Can
        [119, 11, 32],      # Bicycle
        [150, 0, 255],      # Boat
        [0, 60, 100],       # Bus
        [0, 0, 142],        # Car
        [0, 0, 90],         # Caravan
        [0, 0, 230],        # Motorcycle
        [0, 80, 100],       # On Rails
        [128, 64, 64],      # Other Vehicle
        [0, 0, 110],        # Trailer
        [0, 0, 70],         # Truck
        [0, 0, 142],        # Vehicle Group
        [0, 0, 192],        # Wheeled Slow
        [170, 170, 170],    # Water Valve
        [32, 32, 32],       # Car Mount
        [111, 74, 0],       # Dynamic
        [120, 10, 10],      # Ego Vehicle
        [81, 0, 81],        # Ground
        [111, 111, 0],      # Static
        [0, 0, 0],          # Unlabeled
    )

    def __init__(self, data_dir: str, crop_size: Optional[tuple[int, int]] = None):
        super().__init__()

        # load samples - Mapillary 구조: split/images/ + split/v2.0/labels/
        samples = []
        image_dir = os.path.join(data_dir, "images")
        label_dir = os.path.join(data_dir, "v2.0", "labels")
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory not found: {label_dir}")
            print("This might be a testing split without labels")
            # testing 폴더의 경우 라벨 없이 진행
            for fname in sorted(os.listdir(image_dir)):
                if fname.lower().endswith('.jpg'):
                    image_path = os.path.join(image_dir, fname)
                    samples.append((image_path, None))
        else:
            # validation/training 폴더의 경우 라벨과 함께
            for fname in sorted(os.listdir(image_dir)):
                if fname.lower().endswith('.jpg'):
                    image_path = os.path.join(image_dir, fname)
                    # .jpg -> .png 변환
                    label_fname = os.path.splitext(fname)[0] + '.png'
                    label_path = os.path.join(label_dir, label_fname)
                    
                    if os.path.exists(label_path):
                        samples.append((image_path, label_path))
                    else:
                        print(f"Warning: Label not found for {fname}")
        
        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        self.samples = samples
        print(f"Found {len(samples)} samples in Mapillary dataset")

        # build transform
        self.transform = transforms.Compose(
            [
                Resize(crop_size),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        
        # 이미지 로드
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # 마스크 로드 (있는 경우)
        if mask_path is not None:
            mask = np.array(Image.open(mask_path), dtype=np.int64)
            # Mapillary는 0-indexed 이므로 그대로 사용
            # 다만 124 이상의 값은 무시
            mask[mask >= len(self.classes)] = -1
        else:
            # testing의 경우 더미 마스크
            mask = np.full(image.shape[:2], -1, dtype=np.int64)

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path if mask_path is not None else "",
            **feed_dict,
        }


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple | list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        if k < len(colors):
            seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="~/dataset/mapillary/validation")
    parser.add_argument("--dataset", type=str, default="mapillary", choices=["cityscapes", "ade20k", "mapillary"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=1)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

    args.path = os.path.expanduser(args.path)
    if args.dataset == "cityscapes":
        dataset = CityscapesDataset(args.path, (args.crop_size, args.crop_size * 2))
    elif args.dataset == "ade20k":
        dataset = ADE20KDataset(args.path, crop_size=args.crop_size)
    elif args.dataset == "mapillary":
        dataset = MapillaryDataset(args.path, (args.crop_size, args.crop_size))
    else:
        raise NotImplementedError
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = create_efficientvit_seg_model(args.model, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    
    interaction = AverageMeter(is_distributed=False)
    union = AverageMeter(is_distributed=False)
    iou = SegIOU(len(dataset.classes))
    
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on {args.dataset}") as t:
            for feed_dict in data_loader:
                images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
                # compute output
                output = model(images)
                # resize the output to match the shape of the mask
                if output.shape[-2:] != mask.shape[-2:]:
                    output = resize(output, size=mask.shape[-2:])
                output = torch.argmax(output, dim=1)
                stats = iou(output, mask)
                interaction.update(stats["i"])
                union.update(stats["u"])

                t.set_postfix(
                    {
                        "mIOU": (interaction.sum / union.sum).cpu().mean().item() * 100,
                        "image_size": list(images.shape[-2:]),
                    }
                )
                t.update()

                if args.save_path is not None:
                    with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
                            pred = output[i].cpu().numpy()
                            raw_image = np.array(Image.open(image_path).convert("RGB"))
                            canvas = get_canvas(raw_image, pred, dataset.class_colors)
                            canvas = Image.fromarray(canvas)
                            canvas.save(os.path.join(args.save_path, f"{idx}.png"))
                            fout.write(f"{idx}:\t{image_path}\n")

    # print(f"mIoU = {(interaction.sum / union.sum).cpu().mean().item() * 100:.3f}")


if __name__ == "__main__":
    main()