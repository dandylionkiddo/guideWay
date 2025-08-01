import os
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
    def __init__(self, path: str, model: nn.Module, data_provider: DataProvider):
        # 타임스탬프를 포함한 고유한 경로 생성
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(os.path.realpath(os.path.expanduser(path)), timestamp)
        self.model = model.cuda()
        self.data_provider = data_provider

        self.ema = None

        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.logs_path = os.path.join(self.path, "logs")
        for path in [self.path, self.checkpoint_path, self.logs_path]:
            os.makedirs(path, exist_ok=True)

        self.best_val = 0.0
        self.start_epoch = 0

    @property
    def network(self) -> nn.Module:
        return self.model.module if is_parallel(self.model) else self.model

    @property
    def eval_network(self) -> nn.Module:
        if self.ema is None:
            model = self.model
        else:
            model = self.ema.shadows
        model = model.module if is_parallel(model) else model
        return model

    def write_log(self, log_str, prefix="valid", print_log=True, mode="a") -> None:
        if is_master():
            fout = open(os.path.join(self.logs_path, f"{prefix}.log"), mode)
            fout.write(log_str + "\n")
            fout.flush()
            fout.close()
            if print_log:
                print(log_str)

    def save_model(
        self,
        checkpoint=None,
        only_state_dict=True,
        epoch=0,
        model_name=None,
    ) -> None:
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

            latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
            model_path = os.path.join(self.checkpoint_path, model_name)
            with open(latest_fname, "w") as _fout:
                _fout.write(model_path + "\n")
            torch.save(checkpoint, model_path)

    def load_model(self, model_fname=None) -> None:
        latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
        actual_model_fname = None

        if os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname_from_file = fin.readline().strip()
                if model_fname_from_file:
                    actual_model_fname = model_fname_from_file
        
        if model_fname is not None:
            # 사용자가 특정 모델 파일을 지정한 경우
            if os.path.exists(model_fname):
                actual_model_fname = model_fname
            elif os.path.exists(os.path.join(self.checkpoint_path, os.path.basename(model_fname))):
                actual_model_fname = os.path.join(self.checkpoint_path, os.path.basename(model_fname))
            else:
                # 지정된 모델 파일이 없으면 기본 체크포인트 시도
                actual_model_fname = os.path.join(self.checkpoint_path, "checkpoint.pt")
        elif actual_model_fname is None:
            # latest.txt도 없고, model_fname도 지정 안된 경우 기본 체크포인트 시도
            actual_model_fname = os.path.join(self.checkpoint_path, "checkpoint.pt")

        if not os.path.exists(actual_model_fname):
            # 로드할 체크포인트 파일이 아예 없는 경우, 조용히 넘어감
            return

        try:
            print(f"=> loading checkpoint {actual_model_fname}")
            checkpoint = load_state_dict_from_file(actual_model_fname, False)
        except Exception as e:
            self.write_log(f"fail to load checkpoint from {actual_model_fname}. Error: {e}")
            return

        # load checkpoint
        self.network.load_state_dict(checkpoint["state_dict"], strict=False)
        log = []
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            self.run_config.update_global_step(self.start_epoch)
            log.append(f"epoch={self.start_epoch - 1}")
        if "best_val" in checkpoint:
            self.best_val = checkpoint["best_val"]
            log.append(f"best_val={self.best_val:.2f}")
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log.append("optimizer")
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            log.append("lr_scheduler")
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
            log.append("ema")
        if "scaler" in checkpoint and self.enable_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])
            log.append("scaler")
        self.write_log("Loaded: " + ", ".join(log))

    """ validate """

    def reset_bn(
        self,
        network: Optional[nn.Module] = None,
        subset_size: int = 16000,
        subset_batch_size: int = 100,
        data_loader=None,
        progress_bar=False,
    ) -> None:
        network = self.network if network is None else network
        if data_loader is None:
            data_loader = []
            for data in self.data_provider.build_sub_train_loader(subset_size, subset_batch_size):
                if isinstance(data, list):
                    data_loader.append(data[0])
                elif isinstance(data, dict):
                    data_loader.append(data["data"])
                elif isinstance(data, torch.Tensor):
                    data_loader.append(data)
                else:
                    raise NotImplementedError

        network.eval()
        reset_bn(
            network,
            data_loader,
            sync=True,
            progress_bar=progress_bar,
        )

    def _validate(self, model, data_loader, epoch) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, model=None, data_loader=None, is_test=True, epoch=0) -> dict[str, Any]:
        model = self.eval_network if model is None else model
        if data_loader is None:
            if is_test:
                data_loader = self.data_provider.test
            else:
                data_loader = self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch)

    def multires_validate(
        self,
        model=None,
        data_loader=None,
        is_test=True,
        epoch=0,
        eval_image_size=None,
    ) -> dict[str, dict[str, Any]]:
        eval_image_size = self.run_config.eval_image_size if eval_image_size is None else eval_image_size
        eval_image_size = self.data_provider.image_size if eval_image_size is None else eval_image_size
        model = self.eval_network if model is None else model

        if not isinstance(eval_image_size, list):
            eval_image_size = [eval_image_size]

        output_dict = {}
        for r in eval_image_size:
            self.data_provider.assign_active_image_size(parse_image_size(r))
            if self.run_config.reset_bn:
                self.reset_bn(
                    network=model,
                    subset_size=self.run_config.reset_bn_size,
                    subset_batch_size=self.run_config.reset_bn_batch_size,
                    progress_bar=True,
                )
            output_dict[f"r{r}"] = self.validate(model, data_loader, is_test, epoch)
        return output_dict

    """ training """

    def prep_for_training(self, run_config: RunConfig, ema_decay: Optional[float] = None, amp="fp32") -> None:
        self.run_config = run_config
        if get_dist_size() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model.cuda(),
                device_ids=[get_dist_local_rank()],
                static_graph=True,
            )

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        if ema_decay is not None:
            self.ema = EMA(self.network, ema_decay)

        # amp
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

    @property
    def enable_amp(self) -> bool:
        return self.amp != "fp32"

    @property
    def amp_dtype(self) -> torch.dtype:
        if self.amp == "fp16":
            return torch.float16
        elif self.amp == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def sync_model(self):
        print("Sync model")
        self.save_model(model_name="sync.pt")
        dist_barrier()
        checkpoint = torch.load(os.path.join(self.checkpoint_path, "sync.pt"), map_location="cpu", weights_only=True)
        dist_barrier()
        if is_master():
            os.remove(os.path.join(self.checkpoint_path, "sync.pt"))
        dist_barrier()

        # load checkpoint
        self.network.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        if "scaler" in checkpoint and self.enable_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def before_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        for key in feed_dict:
            if isinstance(feed_dict[key], torch.Tensor):
                feed_dict[key] = feed_dict[key].cuda()
        return feed_dict

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def after_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        # gradient clip
        if self.run_config.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.run_config.grad_clip)
        # update
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.lr_scheduler.step()
        self.run_config.step()
        # update ema
        if self.ema is not None:
            self.ema.step(self.network, self.run_config.global_step)

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        raise NotImplementedError

    def train_one_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()

        self.data_provider.set_epoch(epoch)

        train_info_dict = self._train_one_epoch(epoch)

        return train_info_dict

    def train(self) -> None:
        raise NotImplementedError
