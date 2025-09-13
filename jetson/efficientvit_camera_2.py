import cv2
import torch
import numpy as np
import time
import argparse
import gc
import psutil
import subprocess
import os
import signal
import atexit
from datetime import datetime
import tensorrt as trt
import json
from collections import deque

# 작업 디렉토리를 부모 폴더(guideWay)로 변경 
import sys 
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
print(f"Working directory changed to: {os.getcwd()}") 

sys.path.insert(0, parent_dir) 

from efficientvit.models.efficientvit.seg import EfficientViTSeg
import torchvision.transforms as transforms

class JetsonOptimizer:
    """젯슨 하드웨어 최적화 클래스"""
    
    @staticmethod
    def set_max_performance():
        """젯슨을 최대 성능 모드로 설정"""
        try:
            print("Setting Jetson to maximum performance mode...")
            
            subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
            print("✓ Power mode set to maximum")
            
            subprocess.run(['sudo', 'jetson_clocks'], check=True)
            print("✓ Clocks maximized")
            
            gpu_gov_path = '/sys/devices/gpu.0/devfreq/17000000.gv11b/governor'
            if os.path.exists(gpu_gov_path):
                subprocess.run(['sudo', 'sh', '-c', f'echo performance > {gpu_gov_path}'], check=True)
                print("✓ GPU governor set to performance")
                
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

class TensorRTInference:
    """TensorRT 추론 엔진 클래스 - 배치 처리 지원"""
    
    def __init__(self, engine_path, max_batch_size=4):
        """TensorRT 엔진 초기화"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.max_batch_size = max_batch_size
        
        # 엔진 로드
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 입출력 텐서 정보를 동적으로 감지
        input_shape = None
        output_shape = None
        input_name = None
        output_name = None
        
        print(f"Engine has {self.engine.num_io_tensors} I/O tensors")
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            
            print(f"  Tensor {i}: {name}, shape={shape}, dtype={dtype}, mode={mode}")
            
            if mode == trt.TensorIOMode.INPUT or name.lower() == 'input':
                input_shape = shape
                input_name = name
                print(f"  → Detected as INPUT tensor")
            
            elif mode == trt.TensorIOMode.OUTPUT or name.lower() == 'output':
                output_shape = shape
                output_name = name
                print(f"  → Detected as OUTPUT tensor")
        
        # 입출력 차원 설정
        if input_shape is not None:
            self.input_shape = tuple(input_shape)
            self.input_name = input_name
            print(f"✓ Input shape: {self.input_shape}")
        else:
            self.input_shape = (1, 3, 512, 512)
            self.input_name = 'input'
            print(f"⚠️ Using default input shape: {self.input_shape}")
        
        if output_shape is not None:
            self.output_shape = tuple(output_shape)
            self.output_name = output_name
            num_classes = output_shape[1] if len(output_shape) >= 2 else 19
            print(f"✓ Output shape: {self.output_shape}")
            print(f"✓ Detected {num_classes} classes in TRT engine")
        else:
            self.output_shape = (1, 19, 64, 64)
            self.output_name = 'output'
            num_classes = 19
            print(f"⚠️ Using default output shape: {self.output_shape}")
        
        # 배치별 CUDA 메모리 미리 할당
        self.batch_buffers = {}
        for batch_size in range(1, max_batch_size + 1):
            input_shape = (batch_size, *self.input_shape[1:])
            output_shape = (batch_size, *self.output_shape[1:])
            
            self.batch_buffers[batch_size] = {
                'input': torch.empty(*input_shape, dtype=torch.float32, device='cuda'),
                'output': torch.empty(*output_shape, dtype=torch.float32, device='cuda')
            }
        
        self.stream = torch.cuda.Stream()
        
        print(f"✓ TensorRT engine loaded with batch support (max_batch={max_batch_size})")
        
    def __call__(self, input_tensor):
        """배치 추론 실행"""
        batch_size = input_tensor.shape[0]
        
        if batch_size not in self.batch_buffers:
            # 동적으로 새 배치 크기 지원
            input_shape = (batch_size, *self.input_shape[1:])
            output_shape = (batch_size, *self.output_shape[1:])
            
            self.batch_buffers[batch_size] = {
                'input': torch.empty(*input_shape, dtype=torch.float32, device='cuda'),
                'output': torch.empty(*output_shape, dtype=torch.float32, device='cuda')
            }
        
        buffers = self.batch_buffers[batch_size]
        
        # 입력 복사
        buffers['input'].copy_(input_tensor)
        
        # 컨텍스트에 텐서 주소 설정
        self.context.set_tensor_address(self.input_name, buffers['input'].data_ptr())
        self.context.set_tensor_address(self.output_name, buffers['output'].data_ptr())
        
        # TensorRT 추론
        success = self.context.execute_async_v3(self.stream.cuda_stream)
        
        if not success:
            raise RuntimeError("TensorRT execution failed")
            
        self.stream.synchronize()
        
        return buffers['output']

class SafeRealTimeCameraInference:
    def __init__(self, model_name="efficientvit_seg_l1", device="cuda", optimize_jetson=True, 
                 mask_mode=False, use_custom_model=True, batch_size=1):
        print(f"\n=== Initializing Safe Real-time Camera Inference ===")
        
        # 배치 처리 설정
        self.batch_size = min(max(1, batch_size), 8)  # 1-8 사이로 제한
        self.frame_buffer = deque(maxlen=self.batch_size)
        self.result_queue = deque()
        
        # 커스텀 모델 사용 플래그
        self.use_custom_model = use_custom_model
        
        # 모델 정보 확인
        model_info = EfficientViTModelManager.get_model_info(model_name)
        if model_info is None:
            print(f"Warning: Unknown model {model_name}")
            EfficientViTModelManager.list_models()
            raise ValueError(f"Model {model_name} not supported")
        
        self.model_name = model_name
        self.model_info = model_info
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mask_mode = mask_mode
        self.use_trt = False
        
        print(f"Selected Model: {model_name}")
        print(f"Model Info: {model_info['description']}")
        print(f"Parameters: {model_info['params']}")
        print(f"Device: {self.device}")
        print(f"Output Mode: {'Mask only' if mask_mode else 'Overlay'}")
        print(f"Using Custom Model: {use_custom_model}")
        print(f"Batch Size: {self.batch_size}")
        
        # 시스템 최적화
        if optimize_jetson:
            JetsonOptimizer.set_max_performance()
        
        # CUDA 최적화 설정
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ CUDA optimizations enabled")
        
        # 커스텀 클래스 정의 로드
        if use_custom_model:
            self.load_custom_classes()
        else:
            # 기본 Cityscapes 클래스 사용
            self.class_names = [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ]
            self.num_classes = len(self.class_names)
            self.class_colors = self.generate_color_palette(self.num_classes)
        
        # 모델 로드
        self.model = self.load_model()
        
        # 전처리 파이프라인
        self.input_size = model_info['input_size']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 종료 플래그
        self.stop_processing = False
        
        # 안전한 종료를 위한 변수
        self.output_writer = None
        self.current_output_path = None
        self.frames_written = 0
        
        # 시그널 핸들러 등록
        self.setup_signal_handlers()
        
        print("✓ Safe real-time inference initialization completed\n")
    
    def setup_signal_handlers(self):
        """안전한 종료를 위한 시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            print(f"\n\nReceived signal {signum}. Safely stopping...")
            self.safe_cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.safe_cleanup)
    
    def safe_cleanup(self):
        """안전한 정리 작업"""
        print("\n=== Performing safe cleanup ===")
        
        self.stop_processing = True
        
        if self.output_writer is not None:
            try:
                self.output_writer.release()
                print(f"✓ Video saved: {self.current_output_path}")
                print(f"✓ Total frames written: {self.frames_written}")
            except Exception as e:
                print(f"Error releasing video writer: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def load_custom_classes(self):
        """커스텀 클래스 정의 로드"""
        json_path = "applications/efficientvit_seg/custom_class_definitions.json"
        
        try:
            with open(json_path, 'r') as f:
                custom_def = json.load(f)
            
            self.class_names = custom_def['classes']
            self.num_classes = len(self.class_names)
            
            # RGB 색상을 numpy 배열로 변환
            self.class_colors = np.array(custom_def['class_colors'], dtype=np.uint8)
            
            print(f"✓ Custom class definition loaded from {json_path}")
            print(f"✓ Number of classes: {self.num_classes}")
            print(f"✓ Classes: {', '.join(self.class_names[:5])}...")
            
        except FileNotFoundError:
            print(f"✗ Custom class definition not found at {json_path}")
            raise
        except Exception as e:
            print(f"✗ Error loading custom class definition: {e}")
            raise
    
    def load_model(self):
        """TensorRT 또는 PyTorch 모델 로드"""
        print(f"모델 로딩 중: {self.model_name}...")
        
        # TensorRT 엔진 경로 확인
        model_suffix = self.model_name.split('_')[-1]
        if self.use_custom_model:
            trt_path = f"jetson/efficientvit_{model_suffix}_custom_fp32_opt.trt"
        else:
            trt_path = f"jetson/efficientvit_{model_suffix}_fp16.trt"
        
        # TensorRT 엔진이 있으면 사용
        if os.path.exists(trt_path):
            print(f"✓ TensorRT 엔진 발견: {trt_path}")
            self.use_trt = True
            return TensorRTInference(trt_path, max_batch_size=self.batch_size)
        
        # TensorRT 엔진이 없으면 PyTorch 모델 사용
        print(f"TensorRT 엔진 없음 ({trt_path}), PyTorch 모델 사용")
        self.use_trt = False
        
        try:
            from efficientvit.seg_model_zoo import create_efficientvit_seg_model, REGISTERED_EFFICIENTVIT_SEG_MODEL
            
            model_mapping = {
                'efficientvit_seg_b0': 'efficientvit-seg-b0',
                'efficientvit_seg_b1': 'efficientvit-seg-b1', 
                'efficientvit_seg_b2': 'efficientvit-seg-b2',
                'efficientvit_seg_b3': 'efficientvit-seg-b3',
                'efficientvit_seg_l1': 'efficientvit-seg-l1',
                'efficientvit_seg_l2': 'efficientvit-seg-l2'
            }
            
            model_name_mapped = model_mapping.get(self.model_name, 'efficientvit-seg-b0')
            
            if self.use_custom_model:
                # 커스텀 모델 사용
                custom_model_path = "ft-0.0005-coloraug.pt"
                
                if os.path.exists(custom_model_path):
                    print(f"✓ 커스텀 모델 발견: {custom_model_path}")
                    
                    model = create_efficientvit_seg_model(
                        name=model_name_mapped,
                        dataset="mapillary",
                        weight_url=custom_model_path,
                        n_classes=self.num_classes
                    )
                    print(f"✓ 커스텀 모델 로드 완료: {custom_model_path}")
                else:
                    print(f"✗ 커스텀 모델을 찾을 수 없음: {custom_model_path}")
                    print("기본 모델로 대체합니다...")
                    
                    # 기본 모델 로드
                    if model_name_mapped in REGISTERED_EFFICIENTVIT_SEG_MODEL:
                        model_builder, norm_eps, registered_path = REGISTERED_EFFICIENTVIT_SEG_MODEL[model_name_mapped]
                        current_dir = os.getcwd()
                        efficientvit_path = os.path.join(current_dir, "efficientvit", registered_path)
                        
                        if os.path.exists(efficientvit_path):
                            model = create_efficientvit_seg_model(
                                name=model_name_mapped,
                                dataset="cityscapes",
                                weight_url=efficientvit_path,
                                n_classes=19
                            )
                        else:
                            model = create_efficientvit_seg_model(
                                name=model_name_mapped,
                                dataset="cityscapes",
                                pretrained=True,
                                weight_url=None,
                                n_classes=19
                            )
            else:
                # 기본 모델 사용
                if model_name_mapped in REGISTERED_EFFICIENTVIT_SEG_MODEL:
                    model_builder, norm_eps, registered_path = REGISTERED_EFFICIENTVIT_SEG_MODEL[model_name_mapped]
                    
                    current_dir = os.getcwd()
                    efficientvit_path = os.path.join(current_dir, "efficientvit", registered_path)
                    
                    if os.path.exists(efficientvit_path):
                        print(f"✓ 로컬 체크포인트 발견: {efficientvit_path}")
                        model = create_efficientvit_seg_model(
                            name=model_name_mapped,
                            dataset="cityscapes",
                            weight_url=efficientvit_path,
                            n_classes=19
                        )
                    else:
                        print(f"온라인에서 모델 다운로드...")
                        model = create_efficientvit_seg_model(
                            name=model_name_mapped,
                            dataset="cityscapes",
                            pretrained=True,
                            weight_url=None,
                            n_classes=19
                        )
                else:
                    model = create_efficientvit_seg_model(
                        name=model_name_mapped,
                        dataset="cityscapes",
                        pretrained=True,
                        n_classes=19
                    )
                
        except Exception as e:
            print(f"EfficientViT 로딩 실패: {e}")
            raise
        
        # 모델 최적화
        model = model.to(self.device)
        model.eval()
        
        # JIT 컴파일 (배치 지원)
        if self.device == "cuda" and hasattr(torch, 'jit') and self.batch_size == 1:
            try:
                dummy_input = torch.randn(1, 3, *self.model_info['input_size']).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                model = torch.jit.trace(model, dummy_input)
                print("✓ TorchScript로 모델 최적화 완료")
            except Exception as e:
                print(f"TorchScript 최적화 실패: {e}")
        
        # 웜업 (배치 크기별)
        print(f"Warming up model with batch size {self.batch_size}...")
        for _ in range(3):
            dummy_input = torch.randn(self.batch_size, 3, *self.model_info['input_size']).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        print("✓ Model warmup completed")
        
        return model
    
    def generate_color_palette(self, num_classes):
        """클래스 수에 맞는 색상 팔레트 생성"""
        predefined_colors = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]
        ]
        
        colors = []
        for i in range(min(num_classes, len(predefined_colors))):
            colors.append(predefined_colors[i])
        
        return np.array(colors, dtype=np.uint8)
    
    def setup_camera(self, width=1280, height=720, fps=60):
        """카메라 설정"""
        print(f"Setting up camera: {width}x{height} @ {fps}FPS")
        
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true sync=false max-buffers=1"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("✗ Camera setup failed")
            return None
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print("✗ Failed to capture test frame")
            cap.release()
            return None
        
        print(f"✓ Camera setup completed: {frame.shape}")
        return cap
    
    def preprocess_frame(self, frame):
        """프레임 전처리"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).unsqueeze(0)
        return input_tensor.to(self.device, non_blocking=True)
    
    def preprocess_batch(self, frames):
        """배치 프레임 전처리"""
        batch_tensors = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb)
            batch_tensors.append(input_tensor)
        
        batch = torch.stack(batch_tensors)
        return batch.to(self.device, non_blocking=True)
    
    def postprocess_output(self, output, original_shape):
        """모델 출력 후처리"""
        if isinstance(output, dict):
            logits = output.get('out') or output.get('seg') or output.get('logits', list(output.values())[0])
        else:
            logits = output
        
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=original_shape[:2], 
            mode='bilinear',
            align_corners=False
        )
        
        pred = torch.argmax(upsampled_logits, dim=1).squeeze()
        return pred.byte().cpu().numpy()
    
    def batch_inference(self, frames):
        """배치 추론 수행"""
        if len(frames) == 0:
            return []
        
        # 배치 전처리
        batch_input = self.preprocess_batch(frames)
        
        # 배치 추론
        with torch.inference_mode():
            batch_output = self.model(batch_input)
        
        # 결과 분리 및 후처리
        results = []
        for i in range(len(frames)):
            if isinstance(batch_output, dict):
                single_output = {k: v[i:i+1] for k, v in batch_output.items()}
            else:
                single_output = batch_output[i:i+1]
            
            seg_mask = self.postprocess_output(single_output, frames[i].shape[:2])
            results.append(seg_mask)
        
        return results
    
    def create_mask_visualization(self, segmentation_mask):
        """마스크 시각화 생성"""
        colored_mask = self.class_colors[segmentation_mask % len(self.class_colors)]
        colored_mask = colored_mask[..., ::-1]
        return np.ascontiguousarray(colored_mask, dtype=np.uint8)
    
    def create_overlay_visualization(self, frame, segmentation_mask, alpha=0.6):
        """오버레이 시각화 생성"""
        colored_mask = self.class_colors[segmentation_mask % len(self.class_colors)]
        colored_mask = colored_mask[..., ::-1]
        overlay = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        return overlay
    
    def start_batch_inference(self, output_path="camera_output.mp4", 
                            width=1280, height=720, fps=30, show_stats=True):
        """배치 처리를 사용한 실시간 추론"""
        
        print(f"\n=== Starting Batch Real-time Inference ===")
        print(f"Using: {'TensorRT' if self.use_trt else 'PyTorch'}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height}")
        print(f"Output FPS: {fps}")
        print(f"Mode: {'Mask only' if self.mask_mode else 'Overlay'}")
        
        cap = self.setup_camera(width, height, fps=60)
        if cap is None:
            print("✗ Camera setup failed")
            return False
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 960, 540)
        
        inference_times = []
        batch_times = []
        frames_written = 0
        frames_processed = 0
        start_time = time.time()
        
        print("Recording... Press 'q' to stop")
        
        try:
            with torch.inference_mode():
                while not self.stop_processing:
                    # 프레임 수집
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # 프레임을 버퍼에 추가
                    self.frame_buffer.append(frame)
                    
                    # 배치가 가득 차거나 타임아웃되면 처리
                    if len(self.frame_buffer) >= self.batch_size:
                        batch_start = time.time()
                        
                        # 현재 배치 가져오기
                        current_batch = list(self.frame_buffer)
                        self.frame_buffer.clear()
                        
                        # 배치 추론
                        inference_start = time.time()
                        batch_results = self.batch_inference(current_batch)
                        inference_time = time.time() - inference_start
                        
                        inference_times.append(inference_time)
                        batch_times.append((time.time() - batch_start, len(current_batch)))
                        
                        # 결과를 큐에 추가
                        for frame, seg_mask in zip(current_batch, batch_results):
                            self.result_queue.append((frame, seg_mask))
                            frames_processed += 1
                        
                    # 결과 큐에서 프레임 처리 및 저장
                    while self.result_queue:
                        frame, segmentation_mask = self.result_queue.popleft()
                        
                        if self.mask_mode:
                            output_frame = self.create_mask_visualization(segmentation_mask)
                        else:
                            output_frame = self.create_overlay_visualization(frame, segmentation_mask)
                        
                        # 통계 표시 (원본 스타일)
                        if show_stats and inference_times:
                            # 현재 FPS 계산 (배치 크기 고려)
                            latest_inference_time = inference_times[-1] if inference_times else 0
                            current_fps = self.batch_size / latest_inference_time if latest_inference_time > 0 else 0
                            
                            # 평균 FPS 계산
                            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                            avg_fps = self.batch_size / avg_inference_time if avg_inference_time > 0 else 0
                            
                            engine_text = "TRT" if self.use_trt else "PyTorch"
                            cv2.putText(output_frame, f"FPS: {current_fps:.1f} | Avg: {avg_fps:.1f} | {engine_text}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(output_frame, f"Model: {self.model_name}", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # 프레임 저장
                        elapsed = time.time() - start_time
                        expected_frames = int(elapsed * fps)
                        
                        while frames_written < expected_frames:
                            out.write(output_frame)
                            frames_written += 1
                        
                        # 디스플레이 (가장 최근 프레임만)
                        display = cv2.resize(output_frame, (960, 540))
                        cv2.putText(display, f"Recording: {elapsed:.1f}s | Frames: {frames_written}", 
                                (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow('Camera', display)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # 주기적 상태 출력
                    if frames_processed % 100 == 0 and frames_processed > 0:
                        if inference_times:
                            avg_inference = sum(inference_times) / len(inference_times)
                            batch_fps = self.batch_size / avg_inference if avg_inference > 0 else 0
                            print(f"Processed: {frames_processed} | Saved: {frames_written} | "
                                  f"Batch FPS: {batch_fps:.2f} | Queue: {len(self.result_queue)}")
                        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            self.current_output_path = output_path
            self.frames_written = frames_written
            self.safe_cleanup()
            
            print(f"\n✓ Recording completed")
            print(f"✓ Duration: {time.time() - start_time:.1f}s")
            print(f"✓ Frames processed: {frames_processed}")
            print(f"✓ Frames written: {frames_written}")
            print(f"✓ Saved: {output_path}")
            
            if inference_times:
                avg_inference = sum(inference_times) / len(inference_times)
                batch_fps = self.batch_size / avg_inference if avg_inference > 0 else 0
                print(f"✓ Average batch inference time: {avg_inference*1000:.2f}ms")
                print(f"✓ Average batch FPS: {batch_fps:.2f}")
                print(f"✓ Effective throughput: {frames_processed / (time.time() - start_time):.2f} fps")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Safe Real-time EfficientViT Camera Inference with Batch Processing")
    parser.add_argument("--model", "-m", default="efficientvit_seg_l1", 
                       choices=list(EfficientViTModelManager.AVAILABLE_MODELS.keys()),
                       help="EfficientViT model name")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--width", "-w", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size for inference (1-8)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-optimize", action="store_true", help="Skip Jetson optimization")
    parser.add_argument("--no-stats", action="store_true", help="Hide performance overlay")
    parser.add_argument("--mask-mode", action="store_true", help="Output mask only (default: overlay)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--use-default", action="store_true", help="Use default Cityscapes model instead of custom")
    
    args = parser.parse_args()
    
    if args.list_models:
        EfficientViTModelManager.list_models()
        print("\n=== Output Modes ===")
        print("Default: Overlay mode (original + segmentation)")
        print("--mask-mode: Mask only mode (segmentation only)")
        print("\n=== Model Options ===")
        print("Default: Use custom model (ft-0.0005-coloraug.pt)")
        print("--use-default: Use original Cityscapes model")
        print("\n=== Batch Processing ===")
        print("--batch-size N: Process N frames at once (1-8)")
        print("  Higher batch size = higher throughput but more latency")
        print("  Recommended: 2-3 for Jetson Orin")
        return
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "mask" if args.mask_mode else "overlay"
        model_suffix = "default" if args.use_default else "custom"
        batch_suffix = f"batch{args.batch_size}" if args.batch_size > 1 else ""
        args.output = f"camera_{mode_suffix}_{model_suffix}_{batch_suffix}_{args.model}_{timestamp}.mp4"
    
    try:
        inferencer = SafeRealTimeCameraInference(
            model_name=args.model,
            device=args.device,
            optimize_jetson=not args.no_optimize,
            mask_mode=args.mask_mode,
            use_custom_model=not args.use_default,
            batch_size=args.batch_size
        )
        
        inferencer.start_batch_inference(
            output_path=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            show_stats=not args.no_stats
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()