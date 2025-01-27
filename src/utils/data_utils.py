# deep learning libraries
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# other libraries
import os
import random


def load_data(
    path: str, batch_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function loads the data from mnist dataset. All batches must
    be equal size. The division between train and val must be 0.8-0.2.

    Args:
        path: path to save the datasets.
        batch_size: batch size. Defaults to 128.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """

    # define transforms
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.247, 0.243, 0.261]),
        ]
    )

    train_dataset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(
        path, download=True, transform=transformations, train=True
    )
    test_dataset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(
        path, download=True, transform=transformations, train=False
    )

    train_data, val_data = random_split(train_dataset, [0.8, 0.2])
    train_dataloader: DataLoader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader: DataLoader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_dataloader, val_dataloader, test_dataloader
