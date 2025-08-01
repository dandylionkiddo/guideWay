import os

import torch
import torch.distributed

from efficientvit.models.utils.list import list_mean, list_sum

__all__ = [
    "dist_init",
    "is_dist_initialized",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "sync_tensor",
]


def dist_init() -> None:
    if is_dist_initialized():
        return
    try:
        torch.distributed.init_process_group(backend="nccl")
        assert torch.distributed.is_initialized()
    except Exception:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        print("warning: dist not init")


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def sync_tensor(tensor: torch.Tensor | float, reduce="mean") -> torch.Tensor | list[torch.Tensor]:
    if not is_dist_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list
