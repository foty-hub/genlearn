"""
MNIST data utilities for genlearn.
"""

from genlearn.utils import pil_to_jax, jax_collate
import torchvision
import torch
from torch.utils.data import DataLoader, random_split

__all__ = ["get_dataloaders", "get_mnist_dataloaders", "get_cifar_dataloaders"]


def get_dataloaders(
    trainval,
    val_frac: float = 0.1,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(trainval, [1 - val_frac, val_frac], generator=g)
    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=jax_collate,
    )
    valloader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=jax_collate,
    )

    return trainloader, valloader


def get_mnist_dataloaders(
    val_frac: float = 0.1,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
):
    data_path = "../data/mnist"
    trainval = torchvision.datasets.MNIST(
        data_path, download=True, transform=lambda im: pil_to_jax(im, ismnist=True)
    )
    return get_dataloaders(trainval, val_frac, batch_size, shuffle, seed)


def get_cifar_dataloaders(
    val_frac: float = 0.1,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
):
    data_path = "../data"
    trainval = torchvision.datasets.CIFAR10(
        data_path, download=True, transform=lambda im: pil_to_jax(im, ismnist=False)
    )
    return get_dataloaders(trainval, val_frac, batch_size, shuffle, seed)
