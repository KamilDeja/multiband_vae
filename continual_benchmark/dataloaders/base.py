import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset


def MNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def FashionMNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.FashionMNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR10(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    # normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def ToyDataset(dataroot, skip_normalization=False, train_aug=False):
    print("Calculating toy dataset")
    img_size = 60
    square_size = 10
    duplicates = 1

    def create_square(x, y, size=60, square_size=10):
        img = np.zeros([size, size])
        for i in range(square_size):
            for j in range(square_size):
                if (i == j) | (square_size - i - 1 == j):
                    img[x:x + square_size, y:y + square_size] = 1

        return img, x + square_size // 2, y + square_size // 2

    examples = []
    x_list = []
    y_list = []
    for i in range(img_size - square_size):
        for j in range(img_size - square_size):
            img, x, y = create_square(i, j)
            examples.append(img)
            x_list.append(x)
            y_list.append(y)
    examples = torch.tensor(np.array(examples), dtype=torch.float)
    dataset = TensorDataset(examples.repeat([duplicates, 1, 1]),
                            torch.tensor([x_list, y_list], dtype=torch.float32).view(-1, 2).repeat([duplicates, 1]))
    test_dataset = TensorDataset(examples, torch.tensor([x_list, y_list], dtype=torch.float32).view(-1, 2))
    return dataset, test_dataset
