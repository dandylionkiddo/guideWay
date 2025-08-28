"""
이 스크립트는 EfficientViT-Seg 또는 SegFormer 모델의 학습을 시작하는 메인 진입점입니다.
YAML 설정 파일을 기반으로 학습 환경, 데이터 로더, 모델, 트레이너를 구성하고
전체 학습 파이프라인을 실행합니다.
"""

import argparse
import os
import shutil

import torch

# EfficientViT 프레임워크의 핵심 컴포넌트들을 임포트합니다.
from efficientvit.apps.trainer import RunConfig  # 학습 하이퍼파라미터(LR, 옵티마이저 등) 관리
from efficientvit.apps.setup import (
    setup_exp_config,  # YAML 설정 파일 로드 및 병합
    setup_data_provider,  # 데이터셋 및 데이터 로더 설정
    setup_run_config,  # RunConfig 객체 생성
    init_model,  # 모델 가중치 초기화
)
from efficientvit.apps.utils import get_dist_size  # 분산 학습 시 전체 GPU 개수 확인
from efficientvit.models.efficientvit import seg as models  # Segmentation 모델 아키텍처
from efficientvit.segcore.data_provider import SegDataProvider  # Segmentation 데이터 처리기
from efficientvit.segcore.trainer import SegTrainer  # Segmentation 학습 로직 담당 트레이너
from efficientvit.seg_model_zoo import (
    create_efficientvit_seg_model,
    create_segformer_model,
)  # 모델 생성 및 사전학습 가중치 로드

# `efficientvit.models.efficientvit.seg` 모듈에 정의된 모든 모델 빌더 함수를
# 딕셔너리 형태로 등록합니다. 이를 통해 YAML 설정 파일에서 모델 이름을 키로 사용하여
# 동적으로 모델을 생성할 수 있습니다.
ALL_SEG_MODELS = {name: func for name, func in models.__dict__.items() if name.startswith("efficientvit-seg-")}

# 이 학습 스크립트에서 사용할 수 있는 데이터 프로바이더 클래스 목록입니다.
# 현재는 `SegDataProvider`만 사용합니다.
ALL_SEG_DATA_PROVIDERS = [SegDataProvider]

# 커맨드라인 인자를 파싱하기 위한 ArgumentParser를 설정합니다.
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="실험 설정 YAML 파일 경로")
parser.add_argument("--path", type=str, required=True, help="결과 저장 디렉토리 경로")
parser.add_argument(
    "--arch",
    type=str,
    default="efficientvit",
    choices=["efficientvit", "segformer"],
    help="사용할 모델 아키텍처 선택 (efficientvit 또는 segformer)",
)

def main() -> None:
    """
    메인 학습 함수.

    커맨드라인 인자를 파싱하고, 설정 파일에 따라 데이터, 모델, 트레이너를
    구성한 뒤 학습을 시작합니다.

    - 입력:
        - sys.argv를 통해 커맨드라인 인자(--config, --path, --arch)를 받습니다.
    - 출력:
        - 없음 (None). 학습 완료 후 함수가 종료됩니다.
    """
    # 1. 커맨드라인 인자 파싱
    # `parse_known_args`를 사용하여 정의된 인자 외 추가적인 오버라이드 인자도 받을 수 있습니다.
    args, opt_args = parser.parse_known_args()

    # 2. 실험 환경 설정
    # `setup_exp_config` 함수는 YAML 파일을 로드하고, `opt_args`로 받은 값으로
    # 설정을 덮어써서 최종 실험 설정을 확정합니다.
    exp_config = setup_exp_config(args.config, opt_args=opt_args)

    # 3. 데이터 프로바이더 설정
    # `setup_data_provider`는 `exp_config`의 'data_provider' 섹션을 기반으로
    # 데이터셋 로딩, 전처리, 데이터로더 생성을 담당하는 `SegDataProvider` 객체를 생성합니다.
    # 분산 학습 여부(`is_distributed`)도 전달합니다.
    data_provider = setup_data_provider(
        exp_config,
        data_provider_classes=ALL_SEG_DATA_PROVIDERS,
        is_distributed=get_dist_size() > 1,
    )

    # 4. 학습 실행 설정
    # `setup_run_config`은 `exp_config`의 'run_config' 섹션을 기반으로
    # 학습률, 스케줄러, 옵티마이저 등 학습 실행에 필요한 하이퍼파라미터를 담는
    # `RunConfig` 객체를 생성합니다.
    run_config = setup_run_config(exp_config, RunConfig)

    # 5. 모델 생성
    # 설정 파일에서 모델 이름, 클래스 수, 데이터셋 이름을 가져옵니다.
    model_name = exp_config["model"]["name"]
    n_classes = exp_config.get("data_provider", {}).get("n_classes")

    if args.arch == "efficientvit":
        dataset_name = exp_config.get("data_provider", {}).get("dataset")
        model = create_efficientvit_seg_model(
            model_name,
            dataset=dataset_name,
            pretrained=False,
            n_classes=n_classes,
        )
    elif args.arch == "segformer":
        model = create_segformer_model(
            model_name,
            pretrained=False,
            n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # 6. 모델 가중치 초기화
    # `init_model` 함수는 `exp_config`의 'model_init' 섹션에 정의된 방식
    # (예: 랜덤 초기화, 사전학습된 백본 가중치 로드 등)에 따라 모델의 가중치를 초기화합니다.
    init_model(model, **exp_config["model_init"])

    # 7. 트레이너 초기화
    # `SegTrainer` 객체를 생성합니다. 학습 로직, 모델, 데이터 프로바이더, 결과 저장 경로를
    # 모두 여기서 관리합니다.
    trainer = SegTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        # run_config=run_config,  # run_config 전달
    )

    # 8. 학습 시작
    # `prep_for_training`: 옵티마이저, 스케줄러, 분산 학습 설정, AMP(Automatic Mixed Precision) 등 학습 전 준비
    trainer.prep_for_training(run_config, ema_decay=0.9998, amp="fp16")
    # `load_model`: 저장된 체크포인트가 있으면 이어서 학습하기 위해 로드
    trainer.load_model()
    # `train`: 본격적인 학습 루프 시작
    trainer.train()


if __name__ == "__main__":
    main()

