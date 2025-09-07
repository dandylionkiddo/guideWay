# convert_to_onnx.py
import torch
import sys
import os

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from efficientvit.seg_model_zoo import create_efficientvit_seg_model

# 모델 로드
model = create_efficientvit_seg_model(
    name='efficientvit-seg-l1',
    # dataset='cityscapes',
    dataset='mapillary',
    # weight_url='efficientvit/assets/checkpoints/efficientvit_seg/efficientvit_seg_l1_cityscapes.pt',
    weight_url='efficientvit/assets/checkpoints/efficientvit_seg/ft-0.0005-coloraug.pt',
    # n_classes=19
    n_classes=20
)
model.eval().cuda()

# # 모델 출력 확인
# print(f"모델 출력 클래스 수: {model.num_classes if hasattr(model, 'num_classes') else '확인 필요'}") 모델 객체에 num_classes 속성이 명시적으로 저장되어 있지 않을 뿐

# ONNX 변환
dummy_input = torch.randn(1, 3, 512, 512).cuda()

# 먼저 추론 테스트
with torch.no_grad():
    test_output = model(dummy_input)
    if isinstance(test_output, dict):
        print(f"출력 형태: dict with keys {test_output.keys()}")
        for key, val in test_output.items():
            print(f"  {key}: {val.shape}")
    else:
        print(f"출력 형태: {test_output.shape}")

torch.onnx.export(
    model,
    dummy_input,
    # "efficientvit_l1.onnx",
    "efficientvit_l1_custom.onnx",  # 파일명 구분
    input_names=['input'],
    output_names=['output'],
    opset_version=13,
    do_constant_folding=True,
    # export_params=True
    export_params=True,
    dynamic_axes={  # 동적 배치 크기 지원
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)

# print("✓ ONNX 변환 완료: efficientvit_l1.onnx")
print("✓ ONNX 변환 완료: efficientvit_l1_custom.onnx")

# ONNX 모델 검증
import onnx
onnx_model = onnx.load("efficientvit_l1_custom.onnx")
onnx.checker.check_model(onnx_model)
print("✓ ONNX 모델 검증 완료")