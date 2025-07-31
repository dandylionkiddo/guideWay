import os
import time
from copy import deepcopy
from typing import Optional

import torch.backends.cudnn
import torch.distributed
import torch.nn as nn

from efficientvit.apps.data_provider import DataProvider
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import (
    dist_init,
    dump_config,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    init_modules,
    is_master,
    load_config,
    partial_update_config,
    zero_last_gamma,
)
from efficientvit.models.utils import build_kwargs_from_config, load_state_dict_from_file

__all__ = [
    "save_exp_config",
    "setup_dist_env",
    "setup_seed",
    "setup_exp_config",
    "setup_data_provider",
    "setup_run_config",
    "init_model",
]


def save_exp_config(exp_config: dict, path: str, name="config.yaml") -> None:
    """
    실험 설정을 YAML 파일로 저장합니다.
    분산 학습 시, master 프로세스에서만 실행하여 파일 쓰기 충돌을 방지합니다.

    Args:
        exp_config (dict): 저장할 설정 딕셔너리.
        path (str): 저장할 디렉토리 경로.
        name (str, optional): 저장할 파일 이름. Defaults to "config.yaml".
    """
    if not is_master():
        return
    dump_config(exp_config, os.path.join(path, name))


def setup_dist_env(gpu: Optional[str] = None) -> None:
    """
    분산 학습 환경을 설정합니다.

    Args:
        gpu (Optional[str], optional): 사용할 GPU ID. Defaults to None.
    """
    # 사용할 GPU를 환경 변수에 설정
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # 분산 프로세스 그룹 초기화
    if not torch.distributed.is_initialized():
        dist_init()
    # CuDNN 벤치마크 모드를 활성화하여 성능을 최적화
    torch.backends.cudnn.benchmark = True
    # 현재 프로세스에 맞는 로컬 GPU 디바이스를 설정
    torch.cuda.set_device(get_dist_local_rank())


def setup_seed(manual_seed: int, resume: bool) -> None:
    """
    재현성을 위해 랜덤 시드를 설정합니다.

    Args:
        manual_seed (int): 기본 시드 값.
        resume (bool): 학습을 이어서 하는 경우 True.
    """
    # 학습을 이어서 할 경우, 현재 시간을 시드로 사용하여 랜덤성을 부여
    if resume:
        manual_seed = int(time.time())
    # 각 프로세스가 다른 시드를 갖도록 rank를 더해줌
    manual_seed = get_dist_rank() + manual_seed
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


def setup_exp_config(config_path: str, recursive=True, opt_args: Optional[dict] = None) -> dict:
    """
    YAML 설정 파일을 로드하고 병합하여 최종 실험 설정을 구성합니다.

    Args:
        config_path (str): 메인 설정 파일 경로.
        recursive (bool, optional): 상위 디렉토리의 `default.yaml` 파일을 재귀적으로 찾아서 병합할지 여부. Defaults to True.
        opt_args (Optional[dict], optional): 커맨드라인에서 받은 추가 옵션. Defaults to None.

    Returns:
        dict: 최종적으로 구성된 실험 설정.
    """
    if not os.path.isfile(config_path):
        raise ValueError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    fpaths = [config_path]
    # 재귀적으로 상위 디렉토리의 기본 설정(default.yaml)을 찾아 리스트에 추가
    if recursive:
        extension = os.path.splitext(config_path)[1]
        current_path = config_path
        while os.path.dirname(current_path) != current_path:
            current_path = os.path.dirname(current_path)
            fpath = os.path.join(current_path, "default" + extension)
            if os.path.isfile(fpath):
                fpaths.append(fpath)
        fpaths = fpaths[::-1]  # 기본 설정이 먼저 오도록 순서를 뒤집음

    # 기본 설정부터 시작하여 순차적으로 설정을 덮어씀
    default_config = load_config(fpaths[0])
    exp_config = deepcopy(default_config)
    for fpath in fpaths[1:]:
        partial_update_config(exp_config, load_config(fpath))
    
    # 커맨드라인 인자로 받은 추가 옵션으로 설정을 마지막으로 업데이트
    if opt_args is not None:
        partial_update_config(exp_config, opt_args)

    return exp_config


def setup_data_provider(
    exp_config: dict, data_provider_classes: list[type[DataProvider]], is_distributed: bool = True
) -> DataProvider:
    """
    설정 파일에 따라 데이터 프로바이더를 선택하고 구성합니다.

    Args:
        exp_config (dict): 전체 실험 설정.
        data_provider_classes (list[type[DataProvider]]): 사용 가능한 데이터 프로바이더 클래스 리스트.
        is_distributed (bool, optional): 분산 학습 여부. Defaults to True.

    Returns:
        DataProvider: 구성이 완료된 데이터 프로바이더 객체.
    """
    # 데이터 프로바이더 관련 설정만 추출
    dp_config = exp_config["data_provider"]
    
    # 분산 학습 환경에 맞게 복제본 수(num_replicas)와 순위(rank)를 설정
    dp_config["num_replicas"] = get_dist_size() if is_distributed else None
    dp_config["rank"] = get_dist_rank() if is_distributed else None
    
    # 테스트 배치 사이즈 설정 (지정되지 않으면 기본 배치 사이즈의 2배 사용)
    dp_config["test_batch_size"] = dp_config.get("test_batch_size", None)
    dp_config["test_batch_size"] = (
        dp_config["base_batch_size"] * 2 if dp_config["test_batch_size"] is None else dp_config["test_batch_size"]
    )
    dp_config["batch_size"] = dp_config["train_batch_size"] = dp_config["base_batch_size"]

    # 데이터 프로바이더 클래스의 `name` 속성을 키로 하는 룩업 테이블 생성
    data_provider_lookup = {provider.name: provider for provider in data_provider_classes}
    # 설정 파일의 `dataset` 이름에 해당하는 클래스를 룩업 테이블에서 선택
    data_provider_class = data_provider_lookup[dp_config["dataset"]]

    # 선택된 클래스의 `__init__`에 필요한 인자들을 설정에서 자동으로 추출
    data_provider_kwargs = build_kwargs_from_config(dp_config, data_provider_class)
    # 데이터 프로바이더 객체 생성
    data_provider = data_provider_class(**data_provider_kwargs)
    return data_provider


def setup_run_config(exp_config: dict, run_config_cls: type[RunConfig]) -> RunConfig:
    """
    학습 실행 관련 설정을 구성합니다 (예: 학습률, 에폭 수).

    Args:
        exp_config (dict): 전체 실험 설정.
        run_config_cls (type[RunConfig]): 사용할 RunConfig 클래스.

    Returns:
        RunConfig: 구성이 완료된 RunConfig 객체.
    """
    # 분산 학습 시, 전체 GPU 수에 비례하여 학습률을 스케일링 (Linear Scaling Rule)
    exp_config["run_config"]["init_lr"] = exp_config["run_config"]["base_lr"] * get_dist_size()

    # RunConfig 객체 생성
    run_config = run_config_cls(**exp_config["run_config"])

    return run_config


def init_model(
    network: nn.Module,
    init_from: Optional[str] = None,
    backbone_init_from: Optional[str] = None,
    rand_init="trunc_normal",
    last_gamma=None,
) -> None:
    """
    모델의 가중치를 초기화하고, 필요 시 사전 학습된 가중치를 로드합니다.

    Args:
        network (nn.Module): 초기화할 모델.
        init_from (Optional[str], optional): 전체 모델의 사전 학습된 가중치 파일 경로. Defaults to None.
        backbone_init_from (Optional[str], optional): 백본만 사전 학습된 가중치 파일 경로. Defaults to None.
        rand_init (str, optional): 랜덤 초기화 방식. Defaults to "trunc_normal".
        last_gamma (Optional[float], optional): 각 블록의 마지막 Batch Norm의 감마 값을 0으로 설정할지 여부. Defaults to None.
    """
    # 지정된 방식으로 모델의 모든 모듈을 랜덤 초기화
    init_modules(network, init_type=rand_init)
    
    # 각 블록의 마지막 배치 정규화(BN) 레이어의 감마(gamma)를 0으로 초기화하여 학습 안정성을 높임
    if last_gamma is not None:
        zero_last_gamma(network, last_gamma)

    # 사전 학습된 가중치 로드
    if init_from is not None and os.path.isfile(init_from):
        # 전체 모델 가중치 로드
        network.load_state_dict(load_state_dict_from_file(init_from))
        print(f"Loaded init from {init_from}")
    elif backbone_init_from is not None and os.path.isfile(backbone_init_from):
        # 백본 가중치만 로드
        network.backbone.load_state_dict(load_state_dict_from_file(backbone_init_from))
        print(f"Loaded backbone init from {backbone_init_from}")
    else:
        print(f"Random init ({rand_init}) with last gamma {last_gamma}")
