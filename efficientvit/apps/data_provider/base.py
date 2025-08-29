import copy
import warnings
from typing import Any, Optional

import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

from efficientvit.apps.data_provider.random_resolution import RRSController
from efficientvit.models.utils import val2tuple

__all__ = ["parse_image_size", "random_drop_data", "DataProvider"]

def init_rrs_worker(_id):
    """전역 worker_init_fn: Windows spawn 대응"""
    # DataProvider 인스턴스의 _rrs_choice_list는 접근 불가 → 전역 변수에서 가져오도록 함
    global _GLOBAL_RRS_CHOICE_LIST, _GLOBAL_IMAGE_SIZE
    if "_GLOBAL_RRS_CHOICE_LIST" in globals():
        RRSController.CHOICE_LIST = list(_GLOBAL_RRS_CHOICE_LIST)
        RRSController.ACTIVE_SIZE = RRSController.CHOICE_LIST[0]
    elif "_GLOBAL_IMAGE_SIZE" in globals():
        RRSController.CHOICE_LIST = list(_GLOBAL_IMAGE_SIZE)
        RRSController.ACTIVE_SIZE = RRSController.CHOICE_LIST[0]


def parse_image_size(size: int | str) -> tuple[int, int]:
    if isinstance(size, str):
        size = [int(val) for val in size.split("-")]
        return size[0], size[1]
    else:
        return val2tuple(size, 2)


def random_drop_data(dataset, drop_size: int, seed: int, keys=("samples",)):
    g = torch.Generator()
    g.manual_seed(seed)  # set random seed before sampling validation set
    rand_indexes = torch.randperm(len(dataset), generator=g).tolist()

    dropped_indexes = rand_indexes[:drop_size]
    remaining_indexes = rand_indexes[drop_size:]

    dropped_dataset = copy.deepcopy(dataset)
    for key in keys:
        setattr(dropped_dataset, key, [getattr(dropped_dataset, key)[idx] for idx in dropped_indexes])
        setattr(dataset, key, [getattr(dataset, key)[idx] for idx in remaining_indexes])
    return dataset, dropped_dataset


class DataProvider:
    data_keys = ("samples",)
    mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    SUB_SEED = 937162211  # random seed for sampling subset
    VALID_SEED = 2147483647  # random seed for the validation set

    name: str

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: Optional[int],
        valid_size: Optional[int | float],
        n_worker: int,
        image_size: int | list[int] | str | list[str],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
    ):
        warnings.filterwarnings("ignore")
        super().__init__()

        # batch_size & valid_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = self.train_batch_size if test_batch_size is None else test_batch_size
        self.valid_size = valid_size

        # image size
        is_list_of_tuples = isinstance(image_size, list) and all(isinstance(x, (list, tuple)) for x in image_size)

        if is_list_of_tuples:
            # Case 1: RRS with multiple resolutions
            self.image_size = [tuple(size) for size in image_size]
            self.image_size.sort(key=lambda x: x[0] * x[1])
        elif isinstance(image_size, list):
            # Case 2: Single non-square resolution
            self.image_size = tuple(image_size)
        else:
            # Case 3: Single square resolution
            self.image_size = parse_image_size(image_size)

        RRSController.IMAGE_SIZE_LIST = [self.image_size] if not is_list_of_tuples else copy.deepcopy(self.image_size)
        self.active_image_size = RRSController.ACTIVE_SIZE = RRSController.IMAGE_SIZE_LIST[-1]

        # distributed configs
        self.num_replicas = num_replicas
        self.rank = rank

        # build datasets
        train_dataset, val_dataset, test_dataset = self.build_datasets()

        if train_ratio is not None and train_ratio < 1.0:
            assert 0 < train_ratio < 1
            _, train_dataset = random_drop_data(
                train_dataset,
                int(train_ratio * len(train_dataset)),
                self.SUB_SEED,
                self.data_keys,
            )

        # build data loader
        self.train = self.build_dataloader(train_dataset, train_batch_size, n_worker, drop_last=drop_last, train=True)
        self.valid = self.build_dataloader(val_dataset, test_batch_size, n_worker, drop_last=False, train=False)
        self.test = self.build_dataloader(test_dataset, test_batch_size, n_worker, drop_last=False, train=False)
        if self.valid is None:
            self.valid = self.test
        self.sub_train = None

    @property
    def data_shape(self) -> tuple[int, ...]:
        return 3, self.active_image_size[0], self.active_image_size[1]

    def build_valid_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        raise NotImplementedError

    def build_train_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        raise NotImplementedError

    def build_datasets(self) -> tuple[Any, Any, Any]:
        raise NotImplementedError

    def build_dataloader(self, dataset: Optional[Any], batch_size: int, n_worker: int, drop_last: bool, train: bool):
        if dataset is None:
            return None
        use_rrs = isinstance(self.image_size, list) and train
        if use_rrs:
            from efficientvit.apps.data_provider.random_resolution._data_loader import RRSDataLoader

            dataloader_class = RRSDataLoader
            common = dict(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=n_worker,
                pin_memory=True,
                pin_memory_device="cuda",
                persistent_workers=False,
                prefetch_factor=(4 if n_worker > 0 else None),
                drop_last=drop_last,
            )
            common["worker_init_fn"] = init_rrs_worker
            
        else:
            dataloader_class = torch.utils.data.DataLoader
            common = dict(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=n_worker,
                pin_memory=True,
                pin_memory_device="cuda",
                persistent_workers=True,
                prefetch_factor=(4 if n_worker > 0 else None),
                drop_last=drop_last,
            )

        if self.num_replicas is None:
            return dataloader_class(
                shuffle=train,
                **common,
            )
        else:
            sampler = DistributedSampler(dataset, self.num_replicas, self.rank)
            return dataloader_class(
                sampler=sampler,
                **common,
            )

    def set_epoch(self, epoch: int) -> None:
        RRSController.set_epoch(epoch, len(self.train))

        if isinstance(self.image_size, list):  # RRS 모드
            import random
            num_batches = len(self.train)
            candidates = list(self.image_size)
            extra = max(4, num_batches // 10)
            self._rrs_choice_list = [
                random.choice(candidates) for _ in range(num_batches + extra)
            ]
            RRSController.CHOICE_LIST = self._rrs_choice_list
            RRSController.ACTIVE_SIZE = self._rrs_choice_list[0]

            # 🔽 전역 변수에 등록해서 워커도 쓸 수 있게 함
            global _GLOBAL_RRS_CHOICE_LIST
            _GLOBAL_RRS_CHOICE_LIST = self._rrs_choice_list
            global _GLOBAL_IMAGE_SIZE
            _GLOBAL_IMAGE_SIZE = self.image_size

        if isinstance(self.train.sampler, DistributedSampler):
            self.train.sampler.set_epoch(epoch)

    def assign_active_image_size(self, new_size: int | tuple[int, int]) -> None:
        self.active_image_size = val2tuple(new_size, 2)
        new_transform = self.build_valid_transform(self.active_image_size)
        # change the transform of the valid and test set
        self.valid.dataset.transform = self.test.dataset.transform = new_transform

    def sample_val_dataset(self, train_dataset, valid_transform) -> tuple[Any, Any]:
        if self.valid_size is not None:
            if 0 < self.valid_size < 1:
                valid_size = int(self.valid_size * len(train_dataset))
            else:
                assert self.valid_size >= 1
                valid_size = int(self.valid_size)
            train_dataset, val_dataset = random_drop_data(
                train_dataset,
                valid_size,
                self.VALID_SEED,
                self.data_keys,
            )
            val_dataset.transform = valid_transform
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def build_sub_train_loader(self, n_samples: int, batch_size: int) -> Any:
        # used for resetting BN running statistics
        if self.sub_train is None:
            self.sub_train = {}
        if self.active_image_size in self.sub_train:
            return self.sub_train[self.active_image_size]

        # construct dataset and dataloader
        train_dataset = copy.deepcopy(self.train.dataset)
        if n_samples < len(train_dataset):
            _, train_dataset = random_drop_data(
                train_dataset,
                n_samples,
                self.SUB_SEED,
                self.data_keys,
            )
        RRSController.ACTIVE_SIZE = self.active_image_size
        train_dataset.transform = self.build_train_transform(image_size=self.active_image_size)
        data_loader = self.build_dataloader(train_dataset, batch_size, self.train.num_workers, True, False)

        # pre-fetch data
        self.sub_train[self.active_image_size] = [
            data for data in data_loader for _ in range(max(1, n_samples // len(train_dataset)))
        ]

        return self.sub_train[self.active_image_size]
