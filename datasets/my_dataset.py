from typing import Any, Callable, Dict, Optional, Union

from classy_vision.dataset import ClassyDataset, register_dataset
from datasets.beauty_dataset import (
    BeautyDataset,
    SampleType,
)
from classy_vision.dataset.transforms import ClassyTransform, build_transforms


@register_dataset("my_dataset")
class MyDataset(ClassyDataset):
    def __init__(
        self,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: int,
        crop_size: int,
        root: str,
    ) -> None:
        dataset = BeautyDataset(
            root, crop_size, SampleType.TUPLE
        )
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MyDataset":
        assert all(key in config for key in ["root", "crop_size"])

        root = config["root"]
        crop_size = config["crop_size"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        transform = build_transforms(transform_config)
        return cls(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            crop_size,
            root,
        )

