from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_num_classes(dataset_name):
    if dataset_name == "cifar10":
        return 10

    if dataset_name == "cifar100":
        return 100

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_class(dataset_name):
    dataset_classes = {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
    }

    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset_classes[dataset_name]


def resolve_project_path(project_dir, path_text):
    configured_path = Path(path_text)

    if configured_path.is_absolute():
        return str(configured_path)

    return str(project_dir / configured_path)


def create_linear_train_transform(training_config):
    image_size = training_config["image_size"]
    transform_layers = [
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(p=training_config["horizontal_flip_probability"]),
        transforms.ToTensor(),
    ]

    if training_config["normalize_images"]:
        transform_layers.append(transforms.Normalize(training_config["cifar_mean"], training_config["cifar_std"]))

    return transforms.Compose(transform_layers)


def create_eval_transform(training_config):
    transform_layers = [transforms.ToTensor()]

    if training_config["normalize_images"]:
        transform_layers.append(transforms.Normalize(training_config["cifar_mean"], training_config["cifar_std"]))

    return transforms.Compose(transform_layers)


def create_cifar_dataset(dataset_name, dataset_dir, train, transform, download):
    dataset_class = get_dataset_class(dataset_name)
    return dataset_class(root=dataset_dir, train=train, transform=transform, download=download)


def create_dataloader(dataset, batch_size, num_workers, shuffle, device, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def repeat_labels_for_two_views(labels):
    return torch.cat([labels, labels], dim=0)
