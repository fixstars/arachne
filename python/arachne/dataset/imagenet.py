try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torchvision

from arachne.env import Env


def imagenet(split: Literal["train", "val", "test"]) -> torchvision.datasets.ImageNet:
    root_path = Env.DATASET_DIR / "IMAGENET"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return torchvision.datasets.ImageNet(str(root_path), split=split, transform=transform)   
