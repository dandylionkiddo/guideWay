import cv2
import torch
import numpy as np
import time
import argparse
import threading
import gc
import psutil
import subprocess
import os
from datetime import datetime

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

class RealTimeCameraInference:
    def __init__(self, model_name="efficientvit_seg_l1", device="cuda", optimize_jetson=True):
        print(f"\n=== Initializing Real-time Camera Inference ===")
        
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
        
        # 모델 로드
        self.model = self.load_model()
        
        # 클래스 설정 (Cityscapes 19개 클래스)
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
            'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        self.num_classes = len(self.class_names)
        self.class_colors = self.generate_color_palette(self.num_classes)
        
        # 전처리 파이프라인
        self.input_size = model_info['input_size']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 실시간 처리를 위한 변수
        self.is_processing = threading.Lock()
        self.latest_frame = None
        self.frame_available = threading.Event()
        self.stop_processing = threading.Event()
        
        print("✓ Real-time inference initialization completed\n")
    
    def load_model(self):
        """EfficientViT 모델 로드"""
        print(f"모델 로딩 중: {self.model_name}...")
        
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
            
            # 동적 경로 해결
            if model_name_mapped in REGISTERED_EFFICIENTVIT_SEG_MODEL:
                model_builder, norm_eps, registered_path = REGISTERED_EFFICIENTVIT_SEG_MODEL[model_name_mapped]
                
                current_dir = os.getcwd()
                efficientvit_path = os.path.join(current_dir, "efficientvit", registered_path)
                
                print(f"등록된 경로: {registered_path}")
                print(f"실제 확인 경로: {efficientvit_path}")
                
                if os.path.exists(efficientvit_path):
                    print(f"✓ 로컬 체크포인트 발견: {efficientvit_path}")
                    model = create_efficientvit_seg_model(
                        name=model_name_mapped,
                        dataset="cityscapes",
                        weight_url=efficientvit_path,
                        n_classes=19
                    )
                    print(f"✓ 로컬에서 모델 로딩 완료: {model_name_mapped}")
                else:
                    print(f"로컬 파일 없음, 온라인 다운로드 시도...")
                    model = create_efficientvit_seg_model(
                        name=model_name_mapped,
                        dataset="cityscapes",
                        pretrained=True,
                        weight_url=None,
                        n_classes=19
                    )
                    print(f"✓ 온라인에서 모델 로딩 완료: {model_name_mapped}")
            else:
                print(f"등록 정보 없음, 온라인 다운로드...")
                model = create_efficientvit_seg_model(
                    name=model_name_mapped,
                    dataset="cityscapes",
                    pretrained=True,
                    n_classes=19
                )
                print(f"✓ 온라인에서 모델 로딩 완료: {model_name_mapped}")
                
        except Exception as e:
            print(f"EfficientViT 로딩 실패: {e}")
            raise
        
        # 모델 최적화
        model = model.to(self.device)
        model.eval()
        
        # JIT 컴파일
        if self.device == "cuda" and hasattr(torch, 'jit'):
            try:
                dummy_input = torch.randn(1, 3, *self.model_info['input_size']).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                model = torch.jit.trace(model, dummy_input)
                print("✓ TorchScript로 모델 최적화 완료")
            except Exception as e:
                print(f"TorchScript 최적화 실패: {e}")
        
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
        """카메라 설정 - 검증된 파이프라인 사용"""
        print(f"Setting up camera: {width}x{height} @ {fps}FPS")
        
        # 검증된 GStreamer 파이프라인
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("✗ Camera setup failed")
            return None
        
        # 프레임 캡처 테스트
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
    
    def postprocess_output(self, output, original_shape):
        """모델 출력 후처리"""
        with torch.no_grad():
            if isinstance(output, dict):
                for key in ['out', 'seg', 'logits']:
                    if key in output:
                        logits = output[key]
                        break
                else:
                    logits = list(output.values())[0]
            else:
                logits = output
            
            # 원본 크기로 업샘플링
            upsampled_logits = torch.nn.functional.interpolate(
                logits, 
                size=original_shape[:2], 
                mode='bilinear',
                align_corners=False
            )
            
            pred = torch.argmax(upsampled_logits, dim=1).squeeze()
            return pred.cpu().numpy().astype(np.uint8)
    
    def create_mask_visualization(self, segmentation_mask):
        """마스크 시각화 생성"""
        colored_mask = self.class_colors[segmentation_mask % len(self.class_colors)]
        colored_mask = colored_mask[..., ::-1]  # RGB -> BGR
        
        # return colored_mask
        # OpenCV 호환을 위해 연속적인 메모리 레이아웃으로 변환
        return np.ascontiguousarray(colored_mask, dtype=np.uint8)
    
    def camera_capture_thread(self, cap):
        """카메라 캡처 스레드"""
        print("Starting camera capture thread...")
        
        while not self.stop_processing.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                continue
            
            # 추론 중이 아닐 때만 프레임 업데이트
            if not self.is_processing.locked():
                self.latest_frame = frame.copy()
                self.frame_available.set()
            
            time.sleep(0.001)
    
    def inference_thread(self, output_writer, show_stats=True, target_fps=30):
        """추론 처리 스레드 - 실시간 속도 맞춤"""
        print("Starting inference thread...")
        
        inference_times = []
        frame_count = 0
        start_real_time = time.time()
        
        with torch.inference_mode():
            while not self.stop_processing.is_set():
                # 새 프레임 대기
                if not self.frame_available.wait(timeout=1.0):
                    continue
                
                # 추론 중 플래그 설정
                with self.is_processing:
                    if self.latest_frame is None:
                        continue
                    
                    current_frame = self.latest_frame.copy()
                    self.frame_available.clear()
                    
                    # 추론 실행
                    start_time = time.time()
                    input_tensor = self.preprocess_frame(current_frame)
                    output = self.model(input_tensor)
                    segmentation_mask = self.postprocess_output(output, current_frame.shape[:2])
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    # 마스크 시각화
                    mask_output = self.create_mask_visualization(segmentation_mask)
                    
                    # 성능 정보 표시
                    if show_stats:
                        current_fps = 1 / inference_time if inference_time > 0 else 0
                        avg_fps = len(inference_times) / sum(inference_times) if inference_times else 0
                        
                        fps_text = f"Current FPS: {current_fps:.1f} | Avg FPS: {avg_fps:.1f}"
                        model_text = f"Model: {self.model_name}"
                        
                        cv2.putText(mask_output, fps_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(mask_output, model_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 🔥 실시간 속도에 맞춰 프레임 반복 저장
                    current_real_time = time.time() - start_real_time
                    expected_frames = int(current_real_time * target_fps)
                    
                    # 현재까지 저장해야 할 프레임 수만큼 반복 저장
                    while frame_count < expected_frames:
                        # 비디오 저장
                        output_writer.write(mask_output)
                        frame_count += 1
                    
                    # 주기적으로 통계 출력
                    if frame_count % 100 == 0:
                        avg_fps = len(inference_times) / sum(inference_times) if inference_times else 0
                        print(f"Processed {len(inference_times)} inferences, Saved {frame_count} frames, Avg FPS: {avg_fps:.2f}")
                        
                        # 메모리 관리
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
        
    def start_realtime_inference(self, duration_seconds=60, output_path="camera_output.mp4", 
                            width=1280, height=720, fps=30, show_stats=True):
        """실시간 추론 시작 - 캘리브레이션 없이"""
        
        print(f"\n=== Starting Real-time Inference ===")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height}")
        print(f"Output FPS: {fps}")
        
        # 카메라 설정
        cap = self.setup_camera(width, height, fps=60)  # 60FPS 입력
        if cap is None:
            print("✗ Camera setup failed - exiting")
            return False
        
        # 출력 비디오 설정
        # 바로 목표 FPS로 출력 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 스레드 시작
        capture_thread = threading.Thread(target=self.camera_capture_thread, args=(cap,))
        # 녹화 스레드 시작
        inference_thread = threading.Thread(
            target=self.simple_realtime_thread, 
            args=(out, show_stats, fps)
        )
        
        capture_thread.start()
        inference_thread.start()
        
        try:
            print(f"Recording for {duration_seconds} seconds...")
            time.sleep(duration_seconds)
            
        except KeyboardInterrupt:
            print("\nStopping by user request...")
        
        finally:
            # 정리
            print("Cleaning up...")
            self.stop_processing.set()
            
            capture_thread.join(timeout=2.0)
            inference_thread.join(timeout=2.0)
            
            cap.release()
            out.release()
            
            print(f"✓ Video saved to: {output_path}")
            return True

    def simple_realtime_thread(self, output_writer, show_stats, target_fps):
        """단순한 실시간 스레드 - 캘리브레이션 없음"""
        print("Starting simple realtime inference...")
        
        inference_times = []
        start_time = time.time()
        last_mask = None
        frames_written = 0
        
        with torch.inference_mode():
            while not self.stop_processing.is_set():
                # 현재 시간 기준으로 써야 할 프레임 수
                elapsed = time.time() - start_time
                target_frames = int(elapsed * target_fps)
                
                # 새 추론 결과 확인
                if self.frame_available.wait(timeout=0.1):
                    with self.is_processing:
                        if self.latest_frame is not None:
                            current_frame = self.latest_frame.copy()
                            self.frame_available.clear()
                            
                            # 추론
                            inf_start = time.time()
                            input_tensor = self.preprocess_frame(current_frame)
                            output = self.model(input_tensor)
                            mask = self.postprocess_output(output, current_frame.shape[:2])
                            inf_time = time.time() - inf_start
                            inference_times.append(inf_time)
                            
                            # 시각화
                            last_mask = self.create_mask_visualization(mask)
                            
                            # 성능 표시
                            if show_stats and last_mask is not None:
                                current_fps = 1 / inf_time
                                avg_fps = len(inference_times) / sum(inference_times)
                                
                                cv2.putText(last_mask, f"Current: {current_fps:.1f} | Avg: {avg_fps:.1f} FPS", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.putText(last_mask, f"Model: {self.model_name}", 
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 목표 프레임 수만큼 채우기
                while frames_written < target_frames:
                    if last_mask is not None:
                        # 비디오 저장
                        output_writer.write(last_mask)
                    else:
                        # 첫 프레임 없으면 검은 화면
                        black = np.zeros((720, 1280, 3), dtype=np.uint8)
                        output_writer.write(black)
                    frames_written += 1
                
                # 주기적 통계
                if len(inference_times) % 50 == 0 and len(inference_times) > 0:
                    avg_fps = len(inference_times) / sum(inference_times)
                    print(f"Inference count: {len(inference_times)}, Avg FPS: {avg_fps:.2f}, Frames written: {frames_written}")

def main():
    parser = argparse.ArgumentParser(description="Real-time EfficientViT Camera Inference")
    parser.add_argument("--model", "-m", default="efficientvit_seg_l1", 
                       choices=list(EfficientViTModelManager.AVAILABLE_MODELS.keys()),
                       help="EfficientViT model name")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--width", "-w", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-optimize", action="store_true", help="Skip Jetson optimization")
    parser.add_argument("--no-stats", action="store_true", help="Hide performance overlay")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # 모델 목록 출력
    if args.list_models:
        EfficientViTModelManager.list_models()
        return
    
    # 출력 파일명 자동 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"camera_inference_{args.model}_{timestamp}.mp4"
    
    try:
        # 추론 객체 생성
        inferencer = RealTimeCameraInference(
            model_name=args.model,
            device=args.device,
            optimize_jetson=not args.no_optimize
        )
        
        # 실시간 추론 시작
        inferencer.start_realtime_inference(
            duration_seconds=args.duration,
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