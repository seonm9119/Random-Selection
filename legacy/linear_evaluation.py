import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random

from simclr import TransformsSimCLR
from simclr.models import Simclr
from utils import yaml_config_hook

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from simclr.modules import LARS2

import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]


class EvalModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Simclr(hparams)
        self.encoder.load_state_dict(torch.load('runs/Jul11_20-19-07_DESKTOP-CD5F1OB/checkpoint.pth.tar', map_location=hparams.device))
        self.encoder.to(args.device)
        self.encoder.eval()

        n_classes = 100
        hdim = self.encode(torch.ones(10, 3, 32, 32).to(hparams.device)).shape[1]
        model = nn.Linear(hdim, n_classes).to(hparams.device)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def encode(self, x):
        return self.encoder.model(x, out='h')

def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = simclr_model.encode(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)


        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()


    return loss_epoch, accuracy_epoch


def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)

        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )

    else:
        raise NotImplementedError

    args.logistic_batch_size = 512
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    args.logistic_epochs =500
    model = EvalModel(args)
    optimizer = LARS(model.parameters(), lr=0.1, weight_decay=0.0, exclude_from_weight_decay=["batch_normalization", "bias"])
    scheduler = LinearWarmupAndCosineAnneal(optimizer, 0, args.logistic_epochs, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()


    print("### Creating features from pre-trained context model ###")

    (train_X, train_y, test_X, test_y) = get_features( model, train_loader, test_loader, args.device)
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays( train_X, train_y, test_X, test_y, args.logistic_batch_size)


    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, arr_train_loader, model, criterion, optimizer)
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch = test(args, arr_test_loader, model, criterion, optimizer)
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )
    print("seed",seed)
