try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import tensorflow as tf
import tensorflow_datasets as tfds
import torchvision

from arachne.env import Env


def coco_from_tfds(
    version: Literal["2014", "2017"], split: Literal["train", "validation", "test"]
) -> tf.data.Dataset:
    return tfds.load(
        name=f"coco/{version}",
        split=split,
        data_dir=Env.DATASET_DIR / "TFDS",
        shuffle_files=False,
        download=False,
    )


def coco_from_torchvision(
    version: Literal["2014", "2017"], split: Literal["train", "val"]
) -> torchvision.datasets.CocoDetection:
    sub_str = f"{split}{version}"
    coco_path = Env.DATASET_DIR / "COCO"
    root_path = coco_path / sub_str
    annotate_path = coco_path / "annotations" / f"instances_{sub_str}.json"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    return torchvision.datasets.CocoDetection(
        str(root_path), str(annotate_path), transform=transform
    )
