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
        
        # OpenCV 호환을 위해 연속적인 메모리 레이아웃으로 변환
        mask_output = np.ascontiguousarray(colored_mask, dtype=np.uint8)
        
        return mask_output
    
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
    
    def start_realtime_inference(self, duration_seconds=60, output_path="camera_output.mp4", 
                            width=1280, height=720, fps=30, show_stats=True):
        """실시간 추론 속도를 측정하면서 동적으로 FPS 조정하여 정확한 시간 보장"""
        
        print(f"\n=== Starting Adaptive Real-time Inference ===")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height}")
        print(f"Target FPS: {fps} (will adapt to actual inference speed)")
        
        # 카메라 설정
        cap = self.setup_camera(width, height, fps=60)
        if cap is None:
            print("✗ Camera setup failed - exiting")
            return False
        
        # 임시로 높은 FPS로 VideoWriter 생성 (나중에 재인코딩)
        temp_output = f"temp_{output_path}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_writer = cv2.VideoWriter(temp_output, fourcc, 60, (width, height))
        
        # 스레드 시작
        capture_thread = threading.Thread(target=self.camera_capture_thread, args=(cap,))
        inference_thread = threading.Thread(
            target=self.adaptive_realtime_thread, 
            args=(temp_writer, show_stats, duration_seconds)
        )
        
        capture_thread.start()
        inference_thread.start()
        
        try:
            print(f"Recording for {duration_seconds} seconds with adaptive FPS...")
            time.sleep(duration_seconds)
            
        except KeyboardInterrupt:
            print("\nStopping by user request...")
        
        finally:
            print("Cleaning up...")
            self.stop_processing.set()
            
            capture_thread.join(timeout=2.0)
            inference_thread.join(timeout=2.0)
            
            cap.release()
            temp_writer.release()
            
            # 측정된 실제 FPS로 최종 비디오 생성
            self.create_final_video(temp_output, output_path, duration_seconds, width, height)
            
            # 임시 파일 삭제
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            print(f"✓ Video saved to: {output_path}")
            return True

    def adaptive_realtime_thread(self, temp_writer, show_stats, total_duration):
        """실시간으로 추론 속도를 측정하면서 동적 FPS 조정"""
        print("Starting adaptive real-time thread...")
        
        start_time = time.time()
        inference_times = []
        inference_results = []  # (timestamp, frame) 저장
        last_result = None
        
        # FPS 측정 변수
        fps_update_interval = 1.0  # 1초마다 FPS 업데이트
        last_fps_update = start_time
        current_measured_fps = 1.0
        
        with torch.inference_mode():
            while not self.stop_processing.is_set():
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 새로운 추론 실행
                if self.frame_available.wait(timeout=0.01):
                    with self.is_processing:
                        if self.latest_frame is not None:
                            frame = self.latest_frame.copy()
                            self.frame_available.clear()
                            
                            # 추론 실행
                            inf_start = time.time()
                            input_tensor = self.preprocess_frame(frame)
                            output = self.model(input_tensor)
                            mask = self.postprocess_output(output, frame.shape[:2])
                            inf_time = time.time() - inf_start
                            
                            inference_times.append(inf_time)
                            
                            # 시각화
                            result = self.create_mask_visualization(mask)
                            
                            # 통계 표시
                            if show_stats:
                                current_inf_fps = 1 / inf_time if inf_time > 0 else 0
                                avg_inf_fps = len(inference_times) / sum(inference_times) if inference_times else 0
                                
                                cv2.putText(result, f"Current: {current_inf_fps:.1f} | Avg: {avg_inf_fps:.1f} FPS", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.putText(result, f"Measured FPS: {current_measured_fps:.1f} | Time: {elapsed_time:.1f}s", 
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                cv2.putText(result, f"Model: {self.model_name}", 
                                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            last_result = result
                            # 타임스탬프와 함께 결과 저장
                            inference_results.append((elapsed_time, result.copy()))
                
                # 주기적으로 측정된 FPS 업데이트
                if current_time - last_fps_update >= fps_update_interval and inference_times:
                    current_measured_fps = len(inference_times) / sum(inference_times)
                    last_fps_update = current_time
                    print(f"Updated measured FPS: {current_measured_fps:.2f} (total inferences: {len(inference_times)})")
                
                # 임시 파일에 현재 결과 저장 (높은 주기로)
                if last_result is not None:
                    temp_writer.write(last_result)
                
                time.sleep(0.001)  # CPU 부하 감소
        
        # 최종 측정된 FPS 저장
        if inference_times:
            self.final_measured_fps = len(inference_times) / sum(inference_times)
            self.inference_results = inference_results
            print(f"Final measured FPS: {self.final_measured_fps:.2f}")
            print(f"Total inferences: {len(inference_times)}")
            print(f"Total inference results: {len(inference_results)}")
        else:
            self.final_measured_fps = 1.0
            self.inference_results = []

    def create_final_video(self, temp_path, final_path, duration, width, height):
        """측정된 실제 FPS로 최종 비디오 생성"""
        print("Creating final video with measured FPS...")
        
        if not hasattr(self, 'final_measured_fps') or not hasattr(self, 'inference_results'):
            print("Warning: No FPS measurement data, using original file")
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return
        
        measured_fps = self.final_measured_fps
        results = self.inference_results
        
        print(f"Measured FPS: {measured_fps:.2f}")
        print(f"Creating {duration}s video at {measured_fps:.2f} FPS")
        
        # 최종 비디오 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        final_writer = cv2.VideoWriter(final_path, fourcc, measured_fps, (width, height))
        
        if not results:
            print("Warning: No inference results, creating from temp file")
            # 임시 파일에서 복사
            cap = cv2.VideoCapture(temp_path)
            frame_count = 0
            target_frames = int(duration * measured_fps)
            
            while frame_count < target_frames:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 루프
                    continue
                final_writer.write(frame)
                frame_count += 1
            
            cap.release()
        else:
            # 추론 결과를 시간에 맞춰 배치
            total_frames_needed = int(duration * measured_fps)
            frame_interval = duration / total_frames_needed
            
            print(f"Target frames: {total_frames_needed}")
            print(f"Frame interval: {frame_interval:.3f}s")
            
            current_result_idx = 0
            
            for frame_idx in range(total_frames_needed):
                target_time = frame_idx * frame_interval
                
                # 가장 가까운 추론 결과 찾기
                best_result = None
                min_time_diff = float('inf')
                
                for i, (timestamp, result) in enumerate(results):
                    time_diff = abs(timestamp - target_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_result = result
                        current_result_idx = i
                
                if best_result is not None:
                    final_writer.write(best_result)
                else:
                    # 결과가 없으면 검은 화면
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(black_frame, f"No inference at {target_time:.1f}s", 
                            (width//2-150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    final_writer.write(black_frame)
                
                # 진행률 표시
                if frame_idx % (total_frames_needed // 10) == 0:
                    progress = (frame_idx / total_frames_needed) * 100
                    print(f"Final video progress: {progress:.1f}%")
        
        final_writer.release()
        print(f"✓ Final video created: {final_path}")
        print(f"✓ Duration: {duration}s at {measured_fps:.2f} FPS")


# 🔥 main 함수를 클래스 밖으로 이동
def main():
    parser = argparse.ArgumentParser(description="Real-time EfficientViT Camera Inference with Adaptive FPS")
    parser.add_argument("--model", "-m", default="efficientvit_seg_l1", 
                       choices=list(EfficientViTModelManager.AVAILABLE_MODELS.keys()),
                       help="EfficientViT model name")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--width", "-w", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (will adapt to actual speed)")
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
        args.output = f"adaptive_{args.model}_{timestamp}.mp4"
    
    print(f"🎯 Adaptive FPS Mode:")
    print(f"→ Will measure actual inference speed in real-time")
    print(f"→ Final video will use measured FPS for accurate {args.duration}s duration")
    
    try:
        # 추론 객체 생성
        inferencer = RealTimeCameraInference(
            model_name=args.model,
            device=args.device,
            optimize_jetson=not args.no_optimize
        )
        
        # 실행
        success = inferencer.start_realtime_inference(
            duration_seconds=args.duration,
            output_path=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            show_stats=not args.no_stats
        )
        
        if success:
            print(f"\n✅ Recording completed successfully!")
            print(f"✅ Output file: {args.output}")
            print(f"✅ Guaranteed duration: {args.duration} seconds")
        else:
            print("❌ Recording failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()