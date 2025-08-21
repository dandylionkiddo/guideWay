#!/usr/bin/env python3
"""
EfficientViT Jetson Orin Nano Setup
사용법: pip install .
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


class JetsonInstall(install):
    """Jetson Orin Nano 전용 설치 클래스"""
    
    def run(self):
        """설치 실행"""
        print("🚀 Jetson Orin Nano EfficientViT 환경 설정을 시작합니다...")
        
        # 1단계: 시스템 패키지 설치
        self._install_system_packages()
        
        # 2단계: pip 업그레이드
        self._upgrade_pip()
        
        # 3단계: 기존 PyTorch 제거
        self._remove_existing_torch()
        
        # 4단계: 기본 패키지들 설치 (install_requires)
        super().run()
        
        # 5단계: Jetson용 PyTorch 설치
        self._install_jetson_pytorch()
        
        # 6단계: Jetson용 TorchVision 설치
        self._install_jetson_torchvision()
        
        # 7단계: cuSPARSELt 설치
        self._install_cusparselt()
        
        # 8단계: ONNX 패키지 설치
        self._install_onnx_packages()
        
        # 9단계: PyTorch 의존성 패키지들 설치
        self._install_torch_packages()
        
        # 10단계: 환경 변수 설정
        self._setup_environment()
        
        # 11단계: 설치 검증
        self._verify_installation()
        
        print("🎉 설치가 완료되었습니다!")
        print("✅ 환경 설정이 완료되어 바로 사용 가능합니다!")
        print("📋 사용 예시:")
        print("   python3 -c 'import torch; print(torch.cuda.is_available())'")
        print("   python3 -m jetson.efficientvit_inference --help")
    
    def _run_command(self, command, check=True, shell=True):
        """명령어 실행"""
        print(f"실행 중: {command}")
        try:
            result = subprocess.run(command, shell=shell, check=check, 
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ 명령어 실행 실패: {e}")
            if e.stderr:
                print(f"오류: {e.stderr}")
            if check:
                raise
            return e
    
    def _install_system_packages(self):
        """시스템 패키지 설치"""
        print("[1단계] 시스템 패키지 설치")
        
        try:
            self._run_command("sudo apt update")
            self._run_command("sudo apt install -y git cmake build-essential libssl-dev libffi-dev python3-dev")
        except subprocess.CalledProcessError:
            print("⚠️  시스템 패키지 설치를 건너뜁니다. (권한 문제일 수 있음)")
    
    def _upgrade_pip(self):
        """pip 업그레이드"""
        print("[2단계] pip 업그레이드")
        self._run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")
        self._run_command(f'{sys.executable} -m pip install --upgrade "setuptools<70.0.0"')
    
    def _remove_existing_torch(self):
        """기존 PyTorch 제거"""
        print("[3단계] 기존 PyTorch 제거")
        self._run_command(f"{sys.executable} -m pip uninstall torch torchvision torchaudio -y", check=False)
    
    def _install_jetson_pytorch(self):
        """Jetson용 PyTorch 설치"""
        print("[5단계] Jetson용 PyTorch 설치")
        
        pytorch_url = "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
        self._run_command(f"{sys.executable} -m pip install --no-cache {pytorch_url}")
    
    def _install_jetson_torchvision(self):
        """Jetson용 TorchVision 설치"""
        print("[6단계] Jetson용 TorchVision 설치")
        
        torchvision_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
        self._run_command(f"{sys.executable} -m pip install {torchvision_url}")
    
    def _install_cusparselt(self):
        """cuSPARSELt 설치"""
        print("[7단계] cuSPARSELt 설치")
        
        # 이미 설치되어 있는지 확인
        result = self._run_command("ldconfig -p | grep cusparseLt", check=False)
        if result.returncode == 0:
            print("✅ cuSPARSELt가 이미 설치되어 있습니다.")
            return
        
        try:
            print("cuSPARSELt 다운로드 및 설치 중...")
            
            # 다운로드
            cusparselt_url = "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz"
            self._run_command(f"cd /tmp && wget -q {cusparselt_url} -O cusparselt.tar.xz")
            
            # 압축 해제
            self._run_command("cd /tmp && tar xf cusparselt.tar.xz")
            
            # 설치
            self._run_command("sudo cp -a /tmp/libcusparse_lt-*/include/* /usr/local/cuda/include/")
            self._run_command("sudo cp -a /tmp/libcusparse_lt-*/lib/* /usr/local/cuda/lib64/")
            self._run_command("sudo chmod 755 /usr/local/cuda/lib64/libcusparseLt.so*")
            self._run_command("sudo ldconfig")
            
            # 정리
            self._run_command("rm -rf /tmp/libcusparse_lt-* /tmp/cusparselt.tar.xz")
            
            print("✅ cuSPARSELt 설치 완료")
            
        except subprocess.CalledProcessError:
            print("⚠️  cuSPARSELt 자동 설치 실패. 수동 설치가 필요할 수 있습니다.")
    
    def _install_onnx_packages(self):
        """ONNX 패키지 설치"""
        print("[8단계] ONNX 패키지 설치")
        
        # 기존 ONNX 패키지 제거
        self._run_command(f"{sys.executable} -m pip uninstall onnx onnxsim onnxruntime -y", check=False)
        
        # Jetson용 ONNX 설치 시도
        self._run_command(f"{sys.executable} -m pip install onnx --extra-index-url https://developer.download.nvidia.com/compute/redist/", check=False)
        
        # 일반 설치로 폴백
        self._run_command(f"{sys.executable} -m pip install onnx onnxsim onnxruntime", check=False)
    
    def _install_torch_packages(self):
        """PyTorch 의존성 패키지들 설치"""
        print("[9단계] PyTorch 의존성 패키지들 설치")
        
        torch_packages = [
            "torchmetrics", "timm", "torchdiffeq", "torchprofile", "torch-fidelity"
        ]
        
        for package in torch_packages:
            self._run_command(f"{sys.executable} -m pip install {package}", check=False)
        
        # Git 패키지들 설치
        git_packages = [
            "git+https://github.com/alibaba/TinyNeuralNetwork.git",
            "git+https://github.com/facebookresearch/segment-anything.git"
        ]
        
        for git_pkg in git_packages:
            self._run_command(f"{sys.executable} -m pip install {git_pkg}", check=False)
    
    def _setup_environment(self):
        """환경 변수 설정"""
        print("[10단계] 환경 변수 설정")
        
        # 현재 프로세스에 즉시 환경 변수 적용
        cuda_lib_path = '/usr/local/cuda/lib64'
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        
        if cuda_lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}"
            print("✅ 현재 세션에 CUDA 라이브러리 경로 적용 완료")
        
        # ~/.bashrc에도 영구 설정 추가
        bashrc_path = os.path.expanduser("~/.bashrc")
        cuda_path_export = 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
        
        try:
            with open(bashrc_path, 'r') as f:
                content = f.read()
            
            if 'LD_LIBRARY_PATH.*cuda' not in content and cuda_path_export not in content:
                with open(bashrc_path, 'a') as f:
                    f.write(f'\n{cuda_path_export}\n')
                print("✅ CUDA 라이브러리 경로를 ~/.bashrc에 추가했습니다.")
            else:
                print("✅ CUDA 라이브러리 경로가 이미 ~/.bashrc에 설정되어 있습니다.")
        except Exception as e:
            print(f"⚠️  ~/.bashrc 설정 실패: {e}")
        
        # 라이브러리 캐시 업데이트
        try:
            self._run_command("sudo ldconfig", check=False)
        except:
            pass
    
    def _verify_installation(self):
        """설치 검증"""
        print("[11단계] 설치 검증")
        
        # PyTorch 테스트
        test_code = '''
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        print("✅ CUDA operations working!")
    except Exception as e:
        print(f"⚠️  CUDA operation failed: {e}")
else:
    print("❌ CUDA not available")
'''
        
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, check=True,
                                  env=os.environ.copy())
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  PyTorch 검증 실패: {e.stderr}")
        
        # TorchVision 테스트
        torchvision_test = '''
import torch
import torchvision
print(f"TorchVision: {torchvision.__version__}")
try:
    boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8])
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print("✅ NMS operation working!")
    print(f"NMS result: {keep}")
except Exception as e:
    print(f"⚠️  NMS failed: {e}")
'''
        
        try:
            result = subprocess.run([sys.executable, "-c", torchvision_test], 
                                  capture_output=True, text=True, check=True,
                                  env=os.environ.copy())
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  TorchVision 검증 실패: {e.stderr}")


# 패키지 설정
setup(
    name="efficientvit-jetson",
    version="1.0.0",
    description="EfficientViT for Jetson Orin Nano",
    author="GuideWay Team",
    python_requires=">=3.8",
    
    # 일반 의존성들 (PyTorch 비의존성만)
    install_requires=[
        "numpy==1.26.4",
        "einops",
        "scipy",
        "opencv-python",
        "tqdm",
        "omegaconf",
        "ipdb",
        "matplotlib",
        "psutil",
        "huggingface-hub",
        "transformers",
        "diffusers",
        "wandb[media]",
        "pycocotools",
        "lvis",
        "gradio",
        "gradio-clickable-arrow-dropdown",
        "gradio-box-promptable-image", 
        "gradio-point-promptable-image",
        "gradio-sbmp-promptable-image",
    ],
    
    # 커스텀 install 명령 사용
    cmdclass={
        'install': JetsonInstall,
    },
    
    # 패키지 찾기
    packages=find_packages(),
    include_package_data=True,
    
    # 진입점 설정 (선택사항)
    entry_points={
        'console_scripts': [
            'efficientvit-jetson=jetson.efficientvit_inference:main',
        ],
    },
)