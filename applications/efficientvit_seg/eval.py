"""
이 스크립트는 EfficientViT-Seg 모델의 평가를 시작하는 메인 진입점입니다.
YAML 설정 파일을 기반으로 평가 환경, 데이터 로더, 사전 학습된 모델,
그리고 평가자(Evaluator)를 구성하고 전체 평가 파이프라인을 실행합니다.
"""
import argparse
import os

import torch
import yaml

# EfficientViT 프레임워크의 핵심 컴포넌트들을 임포트합니다.
from efficientvit.seg_model_zoo import create_efficientvit_seg_model  # 모델 생성 및 사전학습 가중치 로드
from efficientvit.segcore.data_provider import create_data_loader, create_dataset  # 평가용 데이터셋 및 데이터로더 생성
from efficientvit.segcore.evaluator import Evaluator  # 평가 로직 담당 클래스


def main() -> None:
    """
    메인 평가 함수.

    커맨드라인 인자(--config)로 평가 설정 파일의 경로를 받아,
    그에 따라 데이터, 모델, 평가자를 구성한 뒤 평가를 실행하고 결과를 출력합니다.

    - 입력:
        - sys.argv를 통해 커맨드라인 인자(--config)를 받습니다.
    - 출력:
        - 없음 (None). 평가 완료 후 함수가 종료됩니다.
    """
    # 1. 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="평가 설정 YAML 파일 경로"
    )
    args = parser.parse_args()

    # 2. 평가 설정 파일 로드
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 3. GPU 환경 설정
    gpu_list = config["runtime"].get("gpu", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print(f"Using GPU(s): {gpu_list}")

    # 4. 데이터셋 및 데이터로더 생성
    # `create_dataset`과 `create_data_loader`는 `data_provider.py`에 정의된 팩토리 함수로,
    # 설정 파일의 내용에 따라 적절한 평가용 데이터셋과 데이터로더를 반환합니다.
    dataset = create_dataset(config)
    data_loader = create_data_loader(dataset, config)
    print(f"Loaded dataset {config['dataset']['name']} with {len(dataset)} images.")

    # 5. 모델 생성 및 가중치 로드
    # `create_efficientvit_seg_model` 함수를 통해 모델 아키텍처를 생성하고,
    # `weight_url`에 지정된 경로에서 사전 학습된 가중치를 불러옵니다.
    model = create_efficientvit_seg_model(
        config["model"].get("name"),
        dataset=config["dataset"].get("name"),
        weight_url=config["model"].get("weight_url"),
        n_classes=config["model"].get("n_classes"),
    )
    # 모델을 GPU로 이동시키고, 여러 GPU를 사용할 수 있도록 DataParallel로 감쌉니다.
    model = torch.nn.DataParallel(model).cuda()
    # 모델을 평가 모드(evaluation mode)로 설정합니다. (e.g., Dropout, BatchNorm 비활성화)
    model.eval()
    print(f"Loaded model {config['model']['name']}.")

    # 6. 평가자(Evaluator) 생성 및 평가 실행
    # `Evaluator`는 실제 평가 로직을 담고 있는 클래스입니다.
    evaluator = Evaluator(model, data_loader, config)
    results = evaluator.evaluate()

    # 7. 결과 요약 출력
    print("\n--- Evaluation Summary ---")
    if "mIOU" in results:
        print(f"  mIOU: {results['mIOU']:.3f}%")
    if "fps" in results:
        print(f"  Inference FPS: {results['fps']:.2f}")
    
    save_path = config.get("save_path")
    if save_path:
        # `evaluator.py`에서 생성된 결과 폴더 경로를 알려줍니다.
        print(f"\nResults saved to the latest sub-directory in: {save_path}")
    print("------------------------")


if __name__ == "__main__":
    main()
