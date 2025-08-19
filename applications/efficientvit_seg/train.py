
import argparse
import os
import shutil

import torch

# EfficientViT 관련 유틸리티 및 클래스 임포트
from efficientvit.apps.trainer import RunConfig
from efficientvit.apps.setup import setup_exp_config, setup_data_provider, setup_run_config, init_model
from efficientvit.apps.utils import get_dist_size
# `seg.py`에 있는 모든 모델을 동적으로 사용하기 위해 모듈을 임포트
from efficientvit.models.efficientvit import seg as models
from efficientvit.segcore.data_provider import SegDataProvider
from efficientvit.segcore.trainer import SegTrainer
from efficientvit.seg_model_zoo import create_efficientvit_seg_model

# `seg.py`에 정의된 모든 모델 생성 함수를 딕셔너리로 만듭니다.
# 이제 설정 파일에서 이름만으로 모델을 선택할 수 있습니다.
ALL_SEG_MODELS = {
    name: func for name, func in models.__dict__.items() if name.startswith("efficientvit-seg-")
}

# 사용할 세그멘테이션 데이터 프로바이더를 리스트에 지정합니다.
ALL_SEG_DATA_PROVIDERS = [SegDataProvider]

# 스크립트 실행 시 필요한 인자를 파싱하기 위한 ArgumentParser 설정
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--path", type=str, required=True)

def main():
    # 커맨드라인 인자 파싱
    args, opt_args = parser.parse_known_args()
    # 설정 파일(.yaml)을 읽고 실험 설정을 구성
    exp_config = setup_exp_config(args.config, opt_args=opt_args)

    # 설정 파일에 따라 데이터 프로바이더를 설정
    data_provider = setup_data_provider(
        exp_config,
        data_provider_classes=ALL_SEG_DATA_PROVIDERS,
        is_distributed=get_dist_size() > 1,
    )

    # 학습 실행 관련 설정을 구성
    run_config = setup_run_config(exp_config, RunConfig)

    # 설정 파일에서 사용할 모델의 이름을 가져옵니다.
    model_name = exp_config["model"]["name"]
    # 데이터 프로바이더로부터 n_classes와 dataset 이름을 가져옵니다.
    n_classes = exp_config.get("data_provider", {}).get("n_classes")
    dataset_name = exp_config.get("data_provider", {}).get("dataset")

    # 데이터 프로바이더로부터 데이터셋 이름을 받아와 모델을 생성합니다.
    model = create_efficientvit_seg_model(
        model_name,
        dataset=dataset_name,
        pretrained=False,
        n_classes=n_classes,
    )
    
    # 설정 파일의 `model_init` 섹션을 `init_model` 함수의 인자로 전달합니다.
    # `**`는 딕셔너리를 키워드 인자로 풀어주는 역할을 합니다.
    init_model(model, **exp_config["model_init"])

    # SegTrainer 초기화
    trainer = SegTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
    )

    # 설정 파일(.yaml)을 결과 폴더에 복사
    os.makedirs(args.path, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.path, "config.yaml"))
    print(f"Copied config file to {os.path.join(args.path, 'config.yaml')}")

    # 학습 준비 및 시작
    trainer.prep_for_training(run_config, ema_decay=0.9998, amp="fp16")
    trainer.load_model()
    trainer.train()

if __name__ == "__main__":
    main()

