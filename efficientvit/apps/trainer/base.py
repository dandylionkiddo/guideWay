"""
이 파일은 모든 트레이너의 기반이 되는 `Trainer` 베이스 클래스를 정의합니다.
이 클래스는 특정 태스크에 종속되지 않는 일반적인 학습 로직을 캡슐화합니다.
- 실험 경로 설정 및 폴더 생성
- 모델 저장 및 로드 (체크포인트 관리)
- 로그 작성
- EMA (Exponential Moving Average) 관리
- 학습/검증 루프의 기본 골격 제공
- AMP (Automatic Mixed Precision) 지원

서브클래스(예: `SegTrainer`)는 이 클래스를 상속받아 `_validate`, `run_step`,
`_train_one_epoch`, `train` 등과 같은 추상 메서드를 자신의 태스크에 맞게 구현해야 합니다.
"""
import os
import yaml
from typing import Any, Optional
from datetime import datetime

import torch
import torch.nn as nn

from efficientvit.apps.data_provider import DataProvider, parse_image_size
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import EMA, dist_barrier, get_dist_local_rank, get_dist_size, is_master
from efficientvit.models.nn.norm import reset_bn
from efficientvit.models.utils import is_parallel, load_state_dict_from_file

__all__ = ["Trainer"]


class Trainer:
    """
    모든 트레이너의 베이스 클래스.
    """

    def __init__(self, path: str, model: nn.Module, data_provider: DataProvider) -> None:
        """
        Args:
            path (str): 실험 결과를 저장할 기본 경로. 타임스탬프가 추가된 하위 디렉토리가 생성됩니다.
            model (nn.Module): 학습할 모델.
            data_provider (DataProvider): 데이터를 제공하는 프로바이더.
        """
        # 타임스탬프를 포함한 고유한 실험 경로 생성
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(os.path.realpath(os.path.expanduser(path)), timestamp)
        self.model = model.cuda()
        self.data_provider = data_provider

        self.ema: Optional[EMA] = None

        # 체크포인트와 로그를 저장할 경로 생성
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.logs_path = os.path.join(self.path, "logs")
        for p in [self.path, self.checkpoint_path, self.logs_path]:
            os.makedirs(p, exist_ok=True)

        self.best_val = 0.0
        self.start_epoch = 0

    @property
    def network(self) -> nn.Module:
        """
        분산 학습(`DistributedDataParallel`) 여부에 관계없이 실제 모델 모듈에 접근합니다.
        학습 중인 원본 모델을 반환합니다.
        """
        return self.model.module if is_parallel(self.model) else self.model

    @property
    def eval_network(self) -> nn.Module:
        """
        평가에 사용할 네트워크를 반환합니다.
        EMA가 활성화된 경우 EMA 모델을, 그렇지 않으면 일반 모델을 반환합니다.
        """
        model = self.ema.shadows if self.ema is not None else self.model
        return model.module if is_parallel(model) else model

    def write_log(self, log_str: str, prefix: str = "valid", print_log: bool = True, mode: str = "a") -> None:
        """
        로그 메시지를 파일에 쓰고, 선택적으로 콘솔에도 출력합니다.
        Master 프로세스에서만 실행됩니다.

        Args:
            log_str (str): 기록할 로그 메시지.
            prefix (str, optional): 로그 파일 이름의 접두사. Defaults to "valid".
            print_log (bool, optional): 콘솔 출력 여부. Defaults to True.
            mode (str, optional): 파일 열기 모드 ('a': append, 'w': write). Defaults to "a".
        """
        if is_master():
            with open(os.path.join(self.logs_path, f"{prefix}.log"), mode) as fout:
                fout.write(log_str + "\n")
                fout.flush()
            if print_log:
                print(log_str)

    def save_model(
        self,
        checkpoint: Optional[dict] = None,
        only_state_dict: bool = True,
        epoch: int = 0,
        model_name: Optional[str] = None,
    ) -> None:
        """
        모델의 상태를 체크포인트 파일로 저장합니다.
        Master 프로세스에서만 실행됩니다.

        Args:
            checkpoint (Optional[dict], optional): 저장할 체크포인트 딕셔너리. `None`이면 새로 생성합니다. Defaults to None.
            only_state_dict (bool, optional): `True`이면 모델의 `state_dict`만 저장하고,
                                              `False`이면 옵티마이저, 스케줄러 등 전체 상태를 저장합니다. Defaults to True.
            epoch (int, optional): 현재 에폭 번호.
            model_name (Optional[str], optional): 저장할 모델 파일 이름. `None`이면 'checkpoint.pt'가 됩니다. Defaults to None.
        """
        if is_master():
            if checkpoint is None:
                if only_state_dict:
                    checkpoint = {"state_dict": self.network.state_dict()}
                else:
                    checkpoint = {
                        "state_dict": self.network.state_dict(),
                        "epoch": epoch,
                        "best_val": self.best_val,
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                        "ema": self.ema.state_dict() if self.ema is not None else None,
                        "scaler": self.scaler.state_dict() if self.enable_amp else None,
                    }
            
            model_name = "checkpoint.pt" if model_name is None else model_name
            model_path = os.path.join(self.checkpoint_path, model_name)
            torch.save(checkpoint, model_path)

            # 가장 최근 체크포인트 경로를 latest.txt에 기록
            with open(os.path.join(self.checkpoint_path, "latest.txt"), "w") as fout:
                fout.write(model_path + "\n")

    def load_model(self, model_fname: Optional[str] = None) -> None:
        """
        체크포인트 파일로부터 모델의 상태를 로드합니다.
        옵티마이저, 스케줄러, EMA 등의 상태도 함께 로드하여 학습을 재개할 수 있습니다.

        Args:
            model_fname (Optional[str], optional): 로드할 모델 파일 경로. `None`이면 `latest.txt`에
                                                   기록된 최신 체크포인트를 로드합니다. Defaults to None.
        """
        if model_fname is None:
            latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
            if os.path.exists(latest_fname):
                with open(latest_fname, "r") as fin:
                    model_fname = fin.readline().strip()
            else:
                model_fname = os.path.join(self.checkpoint_path, "checkpoint.pt")

        if not os.path.exists(model_fname):
            self.write_log(f"Checkpoint file not found: {model_fname}", print_log=False)
            return

        try:
            checkpoint = load_state_dict_from_file(model_fname)
            self.write_log(f"=> loading checkpoint {model_fname}")
        except Exception as e:
            self.write_log(f"Failed to load checkpoint from {model_fname}. Error: {e}")
            return

        # 체크포인트에서 각 컴포넌트의 상태를 로드
        if "state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the loaded checkpoint is the state_dict itself
            self.network.load_state_dict(checkpoint)
        
        log = []
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            log.append(f"epoch={self.start_epoch - 1}")
        if "best_val" in checkpoint:
            self.best_val = checkpoint["best_val"]
            log.append(f"best_val={self.best_val:.2f}")
        if hasattr(self, "optimizer") and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log.append("optimizer")
        if hasattr(self, "lr_scheduler") and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            log.append("lr_scheduler")
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
            log.append("ema")
        if hasattr(self, "scaler") and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
            log.append("scaler")
        
        if log:
            self.write_log("Loaded: " + ", ".join(log))

    """ validate """

    def reset_bn(self, network: Optional[nn.Module] = None, **kwargs) -> None:
        """
        모델의 Batch Normalization 통계(running mean, variance)를 재설정합니다.
        주로 다른 해상도로 평가하기 전에 사용됩니다.

        Args:
            network (Optional[nn.Module], optional): BN을 리셋할 모델. `None`이면 `self.network`를 사용. Defaults to None.
        """
        network = self.network if network is None else network
        data_loader = self.data_provider.build_sub_train_loader(**kwargs)
        
        images = []
        for data in data_loader:
            images.append(data[0] if isinstance(data, list) else data["data"])

        reset_bn(network, images, sync=True, progress_bar=True)

    def _validate(self, model: nn.Module, data_loader: Any, epoch: int) -> dict[str, Any]:
        """
        [추상 메서드] 서브클래스에서 반드시 구현해야 합니다.
        주어진 모델과 데이터로더로 실제 검증을 수행하는 로직을 포함합니다.
        """
        raise NotImplementedError

    def validate(self, model: Optional[nn.Module] = None, data_loader: Optional[Any] = None, is_test: bool = True, epoch: int = 0, detailed_analysis: bool = False) -> dict[str, Any]:
        """
        검증을 수행하는 래퍼(wrapper) 함수.
        평가할 모델과 데이터로더를 준비하고 `_validate`를 호출합니다.

        Args:
            model (Optional[nn.Module], optional): 평가할 모델. `None`이면 `self.eval_network` 사용. Defaults to None.
            data_loader (Optional[Any], optional): 사용할 데이터로더. `None`이면 `is_test`에 따라 선택. Defaults to None.
            is_test (bool, optional): `True`이면 테스트 데이터셋, `False`이면 검증 데이터셋 사용. Defaults to True.
            epoch (int, optional): 현재 에폭 번호. Defaults to 0.

        Returns:
            dict[str, Any]: `_validate`의 반환값.
        """
        model = self.eval_network if model is None else model
        if data_loader is None:
            data_loader = self.data_provider.test if is_test else self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch, detailed_analysis)

    def multires_validate(self, eval_image_size: Optional[list[int]] = None, **kwargs) -> dict[str, dict[str, Any]]:
        """
        여러 해상도(multi-resolution)에 대해 검증을 수행합니다.

        Args:
            eval_image_size (Optional[list[int]], optional): 평가할 해상도 리스트. `None`이면 설정 파일 값을 따름. Defaults to None.

        Returns:
            dict[str, dict[str, Any]]: 각 해상도별 검증 결과를 담은 딕셔너리.
        """
        eval_image_size = self.run_config.eval_image_size if eval_image_size is None else eval_image_size
        eval_image_size = self.data_provider.image_size if eval_image_size is None else eval_image_size
        
        if not isinstance(eval_image_size, list):
            eval_image_size = [eval_image_size]

        output_dict = {}
        for r in eval_image_size:
            self.data_provider.assign_active_image_size(parse_image_size(r))
            if self.run_config.reset_bn:
                self.reset_bn(
                    network=self.eval_network,
                    subset_size=self.run_config.reset_bn_size,
                    subset_batch_size=self.run_config.reset_bn_batch_size,
                )
            output_dict[f"r{r}"] = self.validate(**kwargs)
        return output_dict

    """ training """

    def prep_for_training(self, run_config: RunConfig, ema_decay: Optional[float] = None, amp: str = "fp32") -> None:
        """
        본격적인 학습 시작에 앞서 필요한 모든 구성 요소를 준비합니다.
        - 분산 학습 설정 (DDP)
        - 옵티마이저 및 학습률 스케줄러 생성
        - EMA 모델 초기화
        - AMP (Automatic Mixed Precision) 스케일러 초기화

        Args:
            run_config (RunConfig): 학습 실행 관련 설정 객체.
            ema_decay (Optional[float], optional): EMA 감쇠(decay) 값. `None`이면 EMA를 사용하지 않음. Defaults to None.
            amp (str, optional): AMP 타입 ("fp16", "bf16", "fp32"). Defaults to "fp32".
        """
        self.run_config = run_config
        
        # 실행 설정을 yaml 파일로 저장
        config_path = os.path.join(self.logs_path, "run_config.yaml")
        if is_master():
            with open(config_path, "w") as f:
                yaml.dump(self.run_config.__dict__, f, default_flow_style=False, sort_keys=False)
            self.write_log(f"Run config saved to {config_path}", print_log=False)

        if get_dist_size() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model.cuda(),
                device_ids=[get_dist_local_rank()],
                static_graph=True,
            )

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        if ema_decay is not None:
            self.ema = EMA(self.network, ema_decay)

        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

    @property
    def enable_amp(self) -> bool:
        """AMP 활성화 여부를 반환합니다."""
        return self.amp != "fp32"

    @property
    def amp_dtype(self) -> torch.dtype:
        """현재 AMP 설정에 맞는 데이터 타입을 반환합니다."""
        if self.amp == "fp16":
            return torch.float16
        elif self.amp == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def sync_model(self) -> None:
        """(사용되지 않음) 분산 학습 시 모델 동기화를 위한 유틸리티."""
        # ... (implementation details)

    def before_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        """
        [추상 메서드] 서브클래스에서 구현될 수 있습니다.
        학습 스텝 전에 데이터에 특정 변환을 적용합니다. (예: GPU로 이동)
        """
        for key in feed_dict:
            if isinstance(feed_dict[key], torch.Tensor):
                feed_dict[key] = feed_dict[key].cuda()
        return feed_dict

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        """
        [추상 메서드] 서브클래스에서 반드시 구현해야 합니다.
        모델의 순전파, 손실 계산 등 단일 학습 스텝의 핵심 로직을 포함합니다.
        """
        raise NotImplementedError

    def after_step(self) -> None:
        """
        `run_step` 이후에 실행되는 로직.
        그래디언트 클리핑, 옵티마이저 업데이트, 스케줄러 업데이트, EMA 업데이트 등을 수행합니다.
        """
        self.scaler.unscale_(self.optimizer)
        if self.run_config.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.run_config.grad_clip)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.lr_scheduler.step()
        self.run_config.step()
        
        if self.ema is not None:
            self.ema.step(self.network, self.run_config.global_step)

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        """
        [추상 메서드] 서브클래스에서 반드시 구현해야 합니다.
        한 에폭 동안의 학습 루프를 정의합니다.
        """
        raise NotImplementedError

    def train_one_epoch(self, epoch: int) -> dict[str, Any]:
        """
        한 에폭의 학습을 수행하는 래퍼(wrapper) 함수.
        모델을 학습 모드로 설정하고, 데이터로더의 에폭을 설정한 뒤 `_train_one_epoch`를 호출합니다.
        """
        self.model.train()
        self.data_provider.set_epoch(epoch)
        train_info_dict = self._train_one_epoch(epoch)
        return train_info_dict

    def train(self) -> None:
        """
        [추상 메서드] 서브클래스에서 반드시 구현해야 합니다.
        전체 학습 과정을 관리하는 메인 루프를 정의합니다.
        (예: 에폭 반복, 학습, 검증, 모델 저장 등)
        """
        raise NotImplementedError
