import cv2
import torch
import numpy as np
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import threading
import queue
import gc
import psutil
import subprocess
import os
import colorsys

# EfficientViT 관련 임포트
import sys
sys.path.append('.')

from efficientvit.models.efficientvit.seg import EfficientViTSeg
import torchvision.transforms as transforms

class JetsonOptimizer:
    """젯슨 하드웨어 최적화 클래스"""
    
    @staticmethod
    def set_max_performance():
        """젯슨을 최대 성능 모드로 설정"""
        try:
            print("Setting Jetson to maximum performance mode...")
            
            # 최대 성능 모드 설정
            subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
            print("✓ Power mode set to maximum")
            
            # 클럭 최대화
            subprocess.run(['sudo', 'jetson_clocks'], check=True)
            print("✓ Clocks maximized")
            
            # GPU 거버너 설정
            gpu_gov_path = '/sys/devices/gpu.0/devfreq/17000000.gv11b/governor'
            if os.path.exists(gpu_gov_path):
                subprocess.run(['sudo', 'sh', '-c', f'echo performance > {gpu_gov_path}'], check=True)
                print("✓ GPU governor set to performance")
                
            # CPU 거버너 설정
            cpu_count = psutil.cpu_count()
            for i in range(cpu_count):
                cpu_gov_path = f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor'
                if os.path.exists(cpu_gov_path):
                    subprocess.run(['sudo', 'sh', '-c', f'echo performance > {cpu_gov_path}'], check=True)
            print(f"✓ CPU governors set to performance for {cpu_count} cores")
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Some optimization commands failed: {e}")
        except Exception as e:
            print(f"Warning: Optimization setup failed: {e}")
    
    @staticmethod
    def get_system_info():
        """시스템 정보 출력"""
        try:
            # GPU 메모리 정보
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                
            # CPU 정보
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory().total / (1024**3)
            print(f"CPU Cores: {cpu_count}")
            print(f"RAM: {memory:.1f} GB")
            
            # 젯슨 모델 확인
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip()
                    print(f"Jetson Model: {model}")
            except:
                print("Jetson Model: Unknown")
                
        except Exception as e:
            print(f"System info gathering failed: {e}")

class EfficientViTModelManager:
    """EfficientViT 모델 관리 클래스"""
    
    AVAILABLE_MODELS = {
        'efficientvit_seg_b0': {
            'input_size': (512, 512),
            'description': 'Smallest, fastest model - good for real-time',
            'params': '0.7M'
        },
        'efficientvit_seg_b1': {
            'input_size': (512, 512), 
            'description': 'Balanced speed and accuracy',
            'params': '4.8M'
        },
        'efficientvit_seg_b2': {
            'input_size': (512, 512),
            'description': 'Higher accuracy, moderate speed',
            'params': '15.1M'
        },
        'efficientvit_seg_b3': {
            'input_size': (512, 512),
            'description': 'Best accuracy, slower inference',
            'params': '39.8M'
        },
        'efficientvit_seg_l1': {
            'input_size': (512, 512),
            'description': 'Large model - highest accuracy',
            'params': '40.5M'
        },
        'efficientvit_seg_l2': {
            'input_size': (512, 512),
            'description': 'Largest model - best quality',
            'params': '74.8M'
        }
    }
    
    @classmethod
    def list_models(cls):
        """사용 가능한 모델 목록 출력"""
        print("\n=== Available EfficientViT Models ===")
        for model_name, info in cls.AVAILABLE_MODELS.items():
            print(f"{model_name}:")
            print(f"  - Parameters: {info['params']}")
            print(f"  - Description: {info['description']}")
            print(f"  - Input size: {info['input_size']}")
            print()
    
    @classmethod
    def get_model_info(cls, model_name):
        """특정 모델 정보 반환"""
        return cls.AVAILABLE_MODELS.get(model_name, None)

class OptimizedEfficientViTInference:
    def __init__(self, model_name="efficientvit_seg_b0", device="cuda", optimize_jetson=True, 
                 class_mapping="auto"):
        print(f"\n=== Initializing EfficientViT Inference ===")
        
        # 표준 데이터셋 클래스 정의
        self.dataset_classes = {
            'cityscapes': [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ],
            'ade20k': [
                'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
                'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
                'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
                'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
                'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
                'chest_of_drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
                'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
                'pool_table', 'pillow', 'screen_door', 'stairway', 'river', 'bridge',
                'bookcase', 'blind', 'coffee_table', 'toilet', 'flower', 'book',
                'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen_island',
                'computer', 'swivel_chair', 'boat', 'bar', 'arcade_machine',
                'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier',
                'awning', 'streetlight', 'booth', 'television_receiver', 'airplane',
                'dirt_track', 'apparel', 'pole', 'land', 'bannister', 'escalator',
                'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship',
                'fountain', 'conveyer_belt', 'canopy', 'washer', 'plaything',
                'swimming_pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
                'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
                'trade_name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
                'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
                'vase', 'traffic_light', 'tray', 'ashcan', 'fan', 'pier', 'crt_screen',
                'plate', 'monitor', 'bulletin_board', 'shower', 'radiator', 'glass',
                'clock', 'flag'
            ],
            'pascal_voc': [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ],
            'custom_walkway': [
                'sidewalk', 'curb-cut', 'crosswalk', 'road', 'minor-road', 'bike-lane', 'curb', 
                'terrain', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 
                'vegetation', 'sky', 'water', 'sign', 'puddle', 'pothole', 'manhole', 'bench', 
                'pole', 'building', 'wall', 'fence'
            ]
        }
        
        # 클래스 매핑 방식 설정
        self.class_mapping_type = class_mapping
        self.num_classes = None
        self.class_names = None
        self.class_colors = None
        
        # 젯슨 최적화
        if optimize_jetson:
            JetsonOptimizer.set_max_performance()
            JetsonOptimizer.get_system_info()
        
        # 모델 정보 확인
        model_info = EfficientViTModelManager.get_model_info(model_name)
        if model_info is None:
            print(f"Warning: Unknown model {model_name}")
            EfficientViTModelManager.list_models()
            raise ValueError(f"Model {model_name} not supported")
        
        self.model_name = model_name
        self.model_info = model_info
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Selected Model: {model_name}")
        print(f"Model Info: {model_info['description']}")
        print(f"Parameters: {model_info['params']}")
        print(f"Device: {self.device}")
        
        # CUDA 최적화 설정
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ CUDA optimizations enabled")
        
        # 모델 로드
        self.model = self.load_model()
        
        # 모델의 실제 클래스 수 감지 및 클래스 매핑 설정
        self.detect_and_setup_classes()
        
        # 전처리 파이프라인
        input_size = model_info['input_size']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 멀티스레딩을 위한 큐
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        print("✓ Initialization completed\n")
        
    def detect_and_setup_classes(self):
        """모델의 실제 클래스 수를 감지하고 적절한 클래스 매핑 설정"""
        print("Detecting model output classes...")
        
        # 더미 입력으로 모델 출력 형태 확인
        dummy_input = torch.randn(1, 3, *self.model_info['input_size']).to(self.device)
        
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
            
        # 출력 차원에서 클래스 수 추출
        if isinstance(dummy_output, dict):
            # 일부 모델은 딕셔너리 형태로 출력
            for key in ['out', 'seg', 'logits']:
                if key in dummy_output:
                    self.num_classes = dummy_output[key].shape[1]
                    break
        else:
            # 일반적인 텐서 출력
            self.num_classes = dummy_output.shape[1]
        
        print(f"Detected {self.num_classes} output classes from model")
        
        # 클래스 매핑 결정
        if self.class_mapping_type == "auto":
            self.auto_detect_dataset()
        elif self.class_mapping_type in self.dataset_classes:
            self.class_names = self.dataset_classes[self.class_mapping_type]
            print(f"Using {self.class_mapping_type} class mapping")
        else:
            print(f"Unknown class mapping: {self.class_mapping_type}, using auto detection")
            self.auto_detect_dataset()
        
        # 클래스 수 불일치 시 조정
        if len(self.class_names) != self.num_classes:
            print(f"Warning: Model outputs {self.num_classes} classes, but mapping has {len(self.class_names)} classes")
            self.adjust_class_mapping()
        
        # 색상 팔레트 생성
        self.class_colors = self.generate_color_palette(self.num_classes)
        
        print(f"✓ Using {len(self.class_names)} classes: {self.class_names[:5]}{'...' if len(self.class_names) > 5 else ''}")
    
    def auto_detect_dataset(self):
        """클래스 수를 기반으로 데이터셋 자동 감지"""
        if self.num_classes == 19:
            self.class_names = self.dataset_classes['cityscapes']
            print("Auto-detected: Cityscapes dataset (19 classes)")
        elif self.num_classes == 21:
            self.class_names = self.dataset_classes['pascal_voc']
            print("Auto-detected: Pascal VOC dataset (21 classes)")
        elif self.num_classes == 150:
            self.class_names = self.dataset_classes['ade20k']
            print("Auto-detected: ADE20K dataset (150 classes)")
        elif self.num_classes == 27:
            self.class_names = self.dataset_classes['custom_walkway']
            print("Auto-detected: Custom walkway dataset (27 classes)")
        else:
            # 일반적인 클래스 이름 생성
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
            print(f"Unknown dataset: Generated generic class names for {self.num_classes} classes")
    
    def adjust_class_mapping(self):
        """클래스 수 불일치 시 매핑 조정"""
        if len(self.class_names) > self.num_classes:
            # 클래스 이름이 더 많은 경우 자름
            self.class_names = self.class_names[:self.num_classes]
            print(f"Truncated class names to {self.num_classes}")
        else:
            # 클래스 이름이 적은 경우 generic 이름 추가
            for i in range(len(self.class_names), self.num_classes):
                self.class_names.append(f'class_{i}')
            print(f"Extended class names to {self.num_classes}")
    
    def generate_color_palette(self, num_classes):
        """클래스 수에 맞는 색상 팔레트 생성"""
        # 미리 정의된 색상들 (주요 클래스용)
        predefined_colors = [
            [128, 64, 128],   # road/sidewalk - 보라
            [244, 35, 232],   # person - 핑크  
            [70, 70, 70],     # building - 진회색
            [102, 102, 156],  # wall - 연보라
            [190, 153, 153],  # fence - 연갈색
            [153, 153, 153],  # pole - 회색
            [250, 170, 30],   # traffic light - 주황
            [220, 220, 0],    # traffic sign - 노랑
            [107, 142, 35],   # vegetation - 올리브
            [152, 251, 152],  # terrain - 연녹색
            [70, 130, 180],   # sky - 하늘색
            [220, 20, 60],    # person - 빨강
            [255, 0, 0],      # rider - 밝은빨강
            [0, 0, 142],      # car - 파랑
            [0, 0, 70],       # truck - 진파랑
            [0, 60, 100],     # bus - 청록
            [0, 80, 100],     # train - 청록
            [0, 0, 230],      # motorcycle - 밝은파랑
            [119, 11, 32],    # bicycle - 적갈색
        ]
        
        colors = []
        
        # 미리 정의된 색상 사용
        for i in range(min(num_classes, len(predefined_colors))):
            colors.append(predefined_colors[i])
        
        # 부족한 색상은 HSV로 자동 생성
        if num_classes > len(predefined_colors):
            import colorsys
            for i in range(len(predefined_colors), num_classes):
                # HSV 색상 공간에서 균등하게 분포된 색상 생성
                hue = (i * 137.508) % 360  # 황금각 사용으로 균등 분포
                saturation = 0.7 + (i % 3) * 0.1  # 0.7-0.9
                value = 0.8 + (i % 2) * 0.2  # 0.8-1.0
                
                rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
                colors.append([int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)])
        
        return np.array(colors, dtype=np.uint8)
            
    def load_model(self):
        """EfficientViT 사전 훈련된 모델 로드 - 올바른 API 사용"""
        print(f"모델 로딩 중: {self.model_name}...")
        
        model = None
        
        # 방법 1: 올바른 EfficientViT API 사용
        try:
            print("EfficientViT seg_model_zoo로 로딩 중...")
            # 올바른 import 방법
            from efficientvit.seg_model_zoo import create_efficientvit_seg_model
            
            # 체크포인트 파일 경로 설정
            checkpoint_path = "efficientvit/assets/checkpoints/efficientvit_seg/efficientvit_seg_b0_cityscapes.pt"
            
            # 모델명 매핑 (참고 코드 기반)
            model_mapping = {
                'efficientvit_seg_b0': 'efficientvit-seg-b0-cityscapes',
                'efficientvit_seg_b1': 'efficientvit-seg-b1-cityscapes', 
                'efficientvit_seg_b2': 'efficientvit-seg-b2-cityscapes',
                'efficientvit_seg_b3': 'efficientvit-seg-b3-cityscapes',
                'efficientvit_seg_l1': 'efficientvit-seg-l1-cityscapes',
                'efficientvit_seg_l2': 'efficientvit-seg-l2-cityscapes'
            }
            
            model_name_mapped = model_mapping.get(self.model_name, 'efficientvit-seg-b0-cityscapes')
            
            # 체크포인트 파일이 있는지 확인하고 로컬에서 로드
            if os.path.exists(checkpoint_path):
                print(f"✓ 체크포인트 파일 발견: {checkpoint_path}")
                model = create_efficientvit_seg_model(
                    name=model_name_mapped,
                    weight_url=checkpoint_path
                )
                print(f"✓ 로컬 체크포인트에서 EfficientViT 모델 로딩 완료: {model_name_mapped}")
            else:
                # 온라인에서 다운로드
                print("온라인에서 모델 다운로드 중...")
                model = create_efficientvit_seg_model(
                    name=model_name_mapped,
                    pretrained=True
                )
                print(f"✓ 온라인에서 EfficientViT 모델 로딩 완료: {model_name_mapped}")
                
        except ImportError as e:
            print(f"EfficientViT import 실패: {e}")
        except Exception as e:
            print(f"EfficientViT 로딩 실패: {e}")
        
        # # 방법 2: HuggingFace SegFormer 대안
        # if model is None:
        #     try:
        #         print("대안으로 HuggingFace에서 SegFormer 로딩 중...")
        #         from transformers import SegformerForSemanticSegmentation
                
        #         model = SegformerForSemanticSegmentation.from_pretrained(
        #             "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        #         )
        #         print("✓ HuggingFace에서 SegFormer 모델 로딩 완료")
                
        #     except Exception as e:
        #         print(f"HuggingFace SegFormer 실패: {e}")
        
        # # 방법 3: 최소 작동 모델 (최후의 수단)
        # if model is None:
        #     print("최소 작동 모델 생성 중...")
            
        #     import torch.nn as nn
        #     import torch.nn.functional as F
            
        #     class MinimalSegmentationModel(nn.Module):
        #         def __init__(self, num_classes=19):
        #             super().__init__()
        #             self.num_classes = num_classes
                    
        #             # 간단한 encoder-decoder 구조
        #             self.encoder = nn.Sequential(
        #                 nn.Conv2d(3, 64, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #                 nn.Conv2d(64, 64, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #                 nn.MaxPool2d(2),
                        
        #                 nn.Conv2d(64, 128, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #                 nn.Conv2d(128, 128, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #                 nn.MaxPool2d(2),
                        
        #                 nn.Conv2d(128, 256, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #             )
                    
        #             # Decoder
        #             self.decoder = nn.Sequential(
        #                 nn.Conv2d(256, 128, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        
        #                 nn.Conv2d(128, 64, 3, padding=1),
        #                 nn.ReLU(inplace=True),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        
        #                 nn.Conv2d(64, num_classes, 1)
        #             )
                
        #         def forward(self, x):
        #             features = self.encoder(x)
        #             output = self.decoder(features)
                    
        #             # 입력 크기로 리사이즈
        #             output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        #             return output
            
        #     model = MinimalSegmentationModel(num_classes=19)
        #     print("✓ 최소 작동 모델 생성 완료")
        #     print("⚠️  경고: 훈련되지 않은 최소 모델을 데모용으로 사용")
        
        # if model is None:
        #     raise RuntimeError("모든 모델 로딩 방법이 실패했습니다. 설치를 확인해주세요.")
        
        # 모델 최적화
        model = model.to(self.device)
        model.eval()
        
        # TensorRT 최적화
        if self.device == "cuda" and hasattr(torch, 'jit'):
            try:
                dummy_input = torch.randn(1, 3, *self.model_info['input_size']).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # JIT 컴파일
                model = torch.jit.trace(model, dummy_input)
                print("✓ TorchScript로 모델 최적화 완료")
            except Exception as e:
                print(f"TorchScript 최적화 실패: {e}")
        
        return model
    
    def preprocess_frame(self, frame):
        """최적화된 프레임 전처리"""
        # BGR to RGB 변환 최적화
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 변환 적용
        input_tensor = self.transform(frame_rgb).unsqueeze(0)
        return input_tensor.to(self.device, non_blocking=True)
    
    def postprocess_output(self, output, original_shape):
        """최적화된 출력 후처리"""
        # GPU에서 직접 처리
        with torch.no_grad():
            # 모델 출력이 딕셔너리인 경우 처리
            if isinstance(output, dict):
                for key in ['out', 'seg', 'logits']:
                    if key in output:
                        logits = output[key]
                        break
                else:
                    # 첫 번째 값 사용
                    logits = list(output.values())[0]
            else:
                logits = output
            
            # 소프트맥스 및 argmax
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).squeeze()
            
            # CPU로 이동하여 리사이즈
            pred_cpu = pred.cpu().numpy().astype(np.uint8)
            
        # 원본 크기로 리사이즈
        pred_resized = cv2.resize(pred_cpu, 
                                (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        return pred_resized
    
    def create_mask_visualization(self, segmentation_mask):
        """마스크만으로 구성된 시각화 생성"""
        # 벡터화된 색상 매핑 (27개 클래스)
        colored_mask = self.class_colors[segmentation_mask % len(self.class_colors)]
        colored_mask = colored_mask[..., ::-1]  # RGB -> BGR 변환
        
        # 클래스별 픽셀 수 계산
        unique, counts = np.unique(segmentation_mask, return_counts=True)
        class_info = {}
        for class_id, count in zip(unique, counts):
            if class_id < len(self.class_names):
                class_info[self.class_names[class_id]] = count
            else:
                class_info[f'unknown_{class_id}'] = count
        
        return colored_mask, class_info
    
    def create_enhanced_overlay(self, frame, segmentation_mask, alpha=0.6):
        """향상된 오버레이 생성 (원본 프레임 + 마스크)"""
        # 벡터화된 색상 매핑
        colored_mask = self.class_colors[segmentation_mask % len(self.class_colors)]
        colored_mask = colored_mask[..., ::-1]  # RGB -> BGR 변환
        
        # 블렌딩
        overlay = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        
        # 클래스별 픽셀 수 계산
        unique, counts = np.unique(segmentation_mask, return_counts=True)
        class_info = {}
        for class_id, count in zip(unique, counts):
            if class_id < len(self.class_names):
                class_info[self.class_names[class_id]] = count
            else:
                class_info[f'unknown_{class_id}'] = count
        
        return overlay, class_info
    
    def save_mask_legend(self, save_dir):
        """클래스별 색상 범례 저장"""
        legend_height = len(self.class_names) * 30 + 50
        legend_width = 400
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        # 제목
        cv2.putText(legend, "Segmentation Classes", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 각 클래스별 색상과 이름
        for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
            y_pos = 60 + i * 25
            
            # 색상 박스
            cv2.rectangle(legend, (10, y_pos - 10), (40, y_pos + 10), 
                         color.tolist(), -1)
            cv2.rectangle(legend, (10, y_pos - 10), (40, y_pos + 10), 
                         (0, 0, 0), 1)
            
            # 클래스 이름
            cv2.putText(legend, f"{i}: {class_name}", (50, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 범례 저장
        legend_path = save_dir / "class_legend.png"
        cv2.imwrite(str(legend_path), legend)
        print(f"✓ Class legend saved to: {legend_path}")
    
    def frame_producer(self, cap, total_frames):
        """프레임 생산자 스레드"""
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                self.frame_queue.put((frame_idx, frame), timeout=1.0)
                frame_idx += 1
            except queue.Full:
                continue
                
        # 종료 신호
        self.frame_queue.put(None)
    
    def inference_worker(self):
        """추론 워커 스레드"""
        while True:
            try:
                item = self.frame_queue.get(timeout=1.0)
                if item is None:
                    self.result_queue.put(None)
                    break
                    
                frame_idx, frame = item
                
                # 추론
                start_time = time.time()
                input_tensor = self.preprocess_frame(frame)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                segmentation_mask = self.postprocess_output(output, frame.shape[:2])
                inference_time = time.time() - start_time
                
                # 결과 전송
                self.result_queue.put((frame_idx, frame, segmentation_mask, inference_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
                continue
    
    def process_video_optimized(self, input_path, output_path, save_frames=False, save_masks=False,
                              show_stats=True, multithreading=True):
        """최적화된 비디오 처리"""
        cap = cv2.VideoCapture(input_path)
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Video Processing Info ===")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f} seconds")
        print(f"Multithreading: {multithreading}")
        
        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 마스크 비디오 출력 설정 (save_masks가 True인 경우)
        mask_out = None
        if save_masks:
            mask_output_path = output_path.replace('.mp4', '_masks.mp4')
            mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))
            print(f"Mask video will be saved to: {mask_output_path}")
        
        # 프레임/마스크 저장 디렉토리
        if save_frames or save_masks:
            if save_frames:
                frames_dir = Path("output_frames")
                frames_dir.mkdir(exist_ok=True)
            if save_masks:
                masks_dir = Path("output_masks")
                masks_dir.mkdir(exist_ok=True)
                # 클래스 범례 저장
                self.save_mask_legend(masks_dir)
        
        # 통계 변수
        inference_times = []
        class_statistics = {}
        
        if multithreading:
            # 멀티스레드 처리
            producer_thread = threading.Thread(target=self.frame_producer, args=(cap, total_frames))
            inference_thread = threading.Thread(target=self.inference_worker)
            
            producer_thread.start()
            inference_thread.start()
            
            processed_frames = 0
            frame_buffer = {}  # 순서 보장을 위한 버퍼
            expected_frame = 0
            
            pbar = tqdm(total=total_frames, desc="Processing")
            
            try:
                while processed_frames < total_frames:
                    try:
                        result = self.result_queue.get(timeout=5.0)
                        if result is None:
                            break
                            
                        frame_idx, frame, segmentation_mask, inference_time = result
                        inference_times.append(inference_time)
                        
                        # 순서대로 처리하기 위해 버퍼에 저장
                        frame_buffer[frame_idx] = (frame, segmentation_mask, inference_time)
                        
                        # 순서대로 출력
                        while expected_frame in frame_buffer:
                            frame, seg_mask, inf_time = frame_buffer.pop(expected_frame)
                            
                            # 오버레이 생성 (원본 + 마스크)
                            overlay, class_info = self.create_enhanced_overlay(frame, seg_mask)
                            
                            # 마스크만 생성 (save_masks 옵션)
                            if save_masks or mask_out:
                                mask_only, _ = self.create_mask_visualization(seg_mask)
                            
                            # 통계 정보 추가
                            for class_name, count in class_info.items():
                                if class_name not in class_statistics:
                                    class_statistics[class_name] = []
                                class_statistics[class_name].append(count)
                            
                            # 성능 정보 표시
                            if show_stats:
                                fps_text = f"FPS: {1/inf_time:.1f}"
                                model_text = f"Model: {self.model_name}"
                                cv2.putText(overlay, fps_text, (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(overlay, model_text, (10, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # 비디오 저장
                            out.write(overlay)
                            if mask_out is not None:
                                mask_out.write(mask_only)
                            
                            # 개별 프레임/마스크 저장
                            if save_frames:
                                cv2.imwrite(str(frames_dir / f"frame_{expected_frame:06d}.jpg"), overlay)
                            if save_masks:
                                cv2.imwrite(str(masks_dir / f"mask_{expected_frame:06d}.png"), mask_only)
                            
                            expected_frame += 1
                            processed_frames += 1
                            pbar.update(1)
                            
                            # 메모리 관리
                            if processed_frames % 30 == 0:
                                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                                gc.collect()
                    
                    except queue.Empty:
                        print("Timeout waiting for results")
                        break
                        
            except KeyboardInterrupt:
                print("\nProcessing interrupted")
            
            finally:
                producer_thread.join(timeout=1.0)
                inference_thread.join(timeout=1.0)
                pbar.close()
        
        else:
            # 단일 스레드 처리 (안정성 우선)
            pbar = tqdm(total=total_frames, desc="Processing")
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 추론
                start_time = time.time()
                input_tensor = self.preprocess_frame(frame)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                segmentation_mask = self.postprocess_output(output, frame.shape[:2])
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 오버레이 생성 (원본 + 마스크)
                overlay, class_info = self.create_enhanced_overlay(frame, segmentation_mask)
                
                # 마스크만 생성 (save_masks 옵션)
                if save_masks or mask_out:
                    mask_only, _ = self.create_mask_visualization(segmentation_mask)
                
                # 통계 수집
                for class_name, count in class_info.items():
                    if class_name not in class_statistics:
                        class_statistics[class_name] = []
                    class_statistics[class_name].append(count)
                
                # 성능 정보 표시
                if show_stats:
                    fps_text = f"FPS: {1/inference_time:.1f}"
                    model_text = f"Model: {self.model_name}"
                    cv2.putText(overlay, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(overlay, model_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 비디오 저장
                out.write(overlay)
                if mask_out is not None:
                    mask_out.write(mask_only)
                
                # 개별 프레임/마스크 저장
                if save_frames:
                    cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), overlay)
                if save_masks:
                    cv2.imwrite(str(masks_dir / f"mask_{frame_idx:06d}.png"), mask_only)
                
                pbar.update(1)
                
                # 메모리 관리
                if frame_idx % 30 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
            
            pbar.close()
        
        # 리소스 정리
        cap.release()
        out.release()
        if mask_out is not None:
            mask_out.release()
        
        # 성능 통계 출력
        self.print_performance_stats(inference_times, class_statistics, total_frames, output_path)
        
        if save_masks:
            print(f"✓ Mask video saved to: {mask_output_path}")
            print(f"✓ Individual masks saved to: output_masks/")
            print(f"✓ Class legend saved to: output_masks/class_legend.png")
    
    def print_performance_stats(self, inference_times, class_statistics, total_frames, output_path):
        """성능 통계 출력"""
        print(f"\n=== Performance Statistics ===")
        print(f"Model: {self.model_name} ({self.model_info['params']})")
        print(f"Total frames processed: {len(inference_times)}")
        
        if inference_times:
            avg_fps = 1 / np.mean(inference_times)
            min_fps = 1 / np.max(inference_times)
            max_fps = 1 / np.min(inference_times)
            
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Min FPS: {min_fps:.2f}")
            print(f"Max FPS: {max_fps:.2f}")
            print(f"Average inference time: {np.mean(inference_times)*1000:.1f}ms")
        
        print(f"\n=== Segmentation Statistics ===")
        for class_name, counts in class_statistics.items():
            if counts:  # 빈 리스트가 아닌 경우만
                avg_pixels = np.mean(counts)
                max_pixels = np.max(counts)
                percentage = (avg_pixels / (512 * 512)) * 100  # 입력 크기 기준
                # 해당 클래스의 색깔 찾기
                color_info = ""
                if class_name in self.class_names:
                    class_index = self.class_names.index(class_name)
                    if class_index < len(self.class_colors):
                        rgb = self.class_colors[class_index]
                        color_name = self.rgb_to_color_name(rgb)
                        color_info = f" ({color_name})"

                print(f"{class_name}: avg {avg_pixels:.0f} pixels ({percentage:.1f}%), max {max_pixels:.0f}{color_info}")
        
        print(f"\nOutput saved to: {output_path}")
        
        # 메모리 정보
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_cached = torch.cuda.memory_reserved() / (1024**3)
            print(f"\nGPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")

    def rgb_to_color_name(self, rgb):
        """RGB 값을 색깔 이름으로 변환"""
        r, g, b = rgb
        
        # Cityscapes 색상 매핑
        color_map = {
            (128, 64, 128): "purple",      # road/sidewalk
            (244, 35, 232): "pink",        # person  
            (70, 70, 70): "dark_gray",     # building
            (102, 102, 156): "light_purple", # wall
            (190, 153, 153): "light_brown", # fence
            (153, 153, 153): "gray",       # pole
            (250, 170, 30): "orange",      # traffic light
            (220, 220, 0): "yellow",       # traffic sign
            (107, 142, 35): "olive",       # vegetation
            (152, 251, 152): "light_green", # terrain
            (70, 130, 180): "sky_blue",    # sky
            (220, 20, 60): "crimson",      # person
            (255, 0, 0): "red",            # rider
            (0, 0, 142): "blue",           # car
            (0, 0, 70): "dark_blue",       # truck
            (0, 60, 100): "teal",          # bus
            (0, 80, 100): "dark_teal",     # train
            (0, 0, 230): "bright_blue",    # motorcycle
            (119, 11, 32): "dark_red",     # bicycle
        }
        
        rgb_tuple = (r, g, b)
        return color_map.get(rgb_tuple, f"rgb({r},{g},{b})")

    # # 1. RGB 값을 색깔 이름으로 변환하는 함수 추가
    # def rgb_to_color_name(self, rgb):
    #     """RGB 값을 색깔 이름으로 변환"""
    #     r, g, b = rgb
        
    #     # 색깔 매핑 딕셔너리
    #     color_map = {
    #         # 기본 색상들
    #         (255, 0, 0): "red",
    #         (0, 255, 0): "green", 
    #         (0, 0, 255): "blue",
    #         (255, 255, 0): "yellow",
    #         (255, 0, 255): "magenta",
    #         (0, 255, 255): "cyan",
    #         (255, 255, 255): "white",
    #         (0, 0, 0): "black",
    #         (128, 128, 128): "gray",
    #         (255, 165, 0): "orange",
    #         (128, 0, 128): "purple",
    #         (255, 192, 203): "pink",
    #         (165, 42, 42): "brown",
    #         (0, 128, 0): "dark_green",
    #         (0, 0, 128): "navy",
    #         (128, 0, 0): "maroon",
            
    #         # Cityscapes 특정 색상들
    #         (128, 64, 128): "purple",      # road/sidewalk
    #         (244, 35, 232): "pink",        # person  
    #         (70, 70, 70): "dark_gray",     # building
    #         (102, 102, 156): "light_purple", # wall
    #         (190, 153, 153): "light_brown", # fence
    #         (153, 153, 153): "gray",       # pole
    #         (250, 170, 30): "orange",      # traffic light
    #         (220, 220, 0): "yellow",       # traffic sign
    #         (107, 142, 35): "olive",       # vegetation
    #         (152, 251, 152): "light_green", # terrain
    #         (70, 130, 180): "sky_blue",    # sky
    #         (220, 20, 60): "crimson",      # person
    #         (0, 0, 142): "blue",           # car
    #         (0, 0, 70): "dark_blue",       # truck
    #         (0, 60, 100): "teal",          # bus
    #         (0, 80, 100): "dark_teal",     # train
    #         (0, 0, 230): "bright_blue",    # motorcycle
    #         (119, 11, 32): "dark_red",     # bicycle
    #     }
        
    #     # 정확히 일치하는 색상 찾기
    #     rgb_tuple = (r, g, b)
    #     if rgb_tuple in color_map:
    #         return color_map[rgb_tuple]
        
    #     # 가장 가까운 색상 찾기
    #     min_distance = float('inf')
    #     closest_color = "unknown"
        
    #     for color_rgb, color_name in color_map.items():
    #         # 유클리드 거리 계산
    #         distance = sum((a - b) ** 2 for a, b in zip(rgb_tuple, color_rgb)) ** 0.5
    #         if distance < min_distance:
    #             min_distance = distance
    #             closest_color = color_name
        
    #     # 거리가 너무 멀면 RGB 값 그대로 표시
    #     if min_distance > 100:
    #         return f"rgb({r},{g},{b})"
        
    #     return closest_color

    # (선택사항) 클래스별 색상 범례도 개선 
    def save_enhanced_mask_legend(self, save_dir):
        """색깔 이름이 포함된 클래스별 색상 범례 저장"""
        legend_height = len(self.class_names) * 35 + 50  # 조금 더 높게
        legend_width = 500  # 조금 더 넓게
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        # 제목
        cv2.putText(legend, "Segmentation Classes", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 각 클래스별 색상과 이름
        for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
            y_pos = 60 + i * 30
            
            # 색상 박스
            cv2.rectangle(legend, (10, y_pos - 12), (40, y_pos + 12), 
                        color.tolist(), -1)
            cv2.rectangle(legend, (10, y_pos - 12), (40, y_pos + 12), 
                        (0, 0, 0), 1)
            
            # 색깔 이름 추가
            color_name = self.rgb_to_color_name(color)
            
            # 클래스 이름 + 색깔 이름
            text = f"{i}: {class_name} ({color_name})"
            cv2.putText(legend, text, (50, y_pos + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        # 범례 저장
        legend_path = save_dir / "enhanced_class_legend.png"
        cv2.imwrite(str(legend_path), legend)
        print(f"✓ Enhanced class legend saved to: {legend_path}")

    # (선택사항) 터미널에서 실시간 색깔 정보 표시 
    def print_color_mapping(self):
        """현재 사용 중인 색깔 매핑 출력"""
        print(f"\n=== Color Mapping for {len(self.class_names)} Classes ===")
        for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
            color_name = self.rgb_to_color_name(color)
            rgb_str = f"RGB({color[0]}, {color[1]}, {color[2]})"
            print(f"{i:2d}: {class_name:15s} -> {color_name:12s} {rgb_str}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Optimized EfficientViT Segmentation for Jetson")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="output_segmented.mp4", help="Output video path")
    parser.add_argument("--model", "-m", default="efficientvit_seg_b0", 
                       choices=list(EfficientViTModelManager.AVAILABLE_MODELS.keys()),
                       help="EfficientViT model name")
    parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save-frames", action="store_true", help="Save individual overlay frames")
    parser.add_argument("--save-masks", action="store_true", help="Save individual mask frames and mask video")
    parser.add_argument("--no-optimize", action="store_true", help="Skip Jetson optimization")
    parser.add_argument("--single-thread", action="store_true", help="Use single thread processing")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--no-stats", action="store_true", help="Hide performance overlay")
    parser.add_argument("--class-mapping", default="auto", 
                       choices=["auto", "cityscapes", "ade20k", "pascal_voc", "custom_walkway"],
                       help="Class mapping dataset (auto: detect from model output)")
    parser.add_argument("--show-classes", action="store_true", help="Show detected classes and exit")
    
    args = parser.parse_args()
    
    # 모델 목록 출력
    if args.list_models:
        EfficientViTModelManager.list_models()
        return
    
    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return
    
    try:
        # 추론 객체 생성
        inferencer = OptimizedEfficientViTInference(
            model_name=args.model, 
            device=args.device,
            optimize_jetson=not args.no_optimize,
            class_mapping=args.class_mapping
        )
        
        # 클래스 정보만 출력하고 종료
        if args.show_classes:
            print(f"\n=== Detected Classes ({len(inferencer.class_names)}) ===")
            for i, class_name in enumerate(inferencer.class_names):
                color = inferencer.class_colors[i]
                color_name = inferencer.rgb_to_color_name(color)
                print(f"{i:3d}: {class_name:20s} RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d}) ({color_name})")
            return
        
        # 비디오 처리
        inferencer.process_video_optimized(
            args.input, 
            args.output, 
            save_frames=args.save_frames,
            save_masks=args.save_masks,
            show_stats=not args.no_stats,
            multithreading=not args.single_thread
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()