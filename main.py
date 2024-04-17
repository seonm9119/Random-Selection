import os
import argparse
import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DataParallel
import torch.distributed as dist


from utils import yaml_config_hook
from simclr import TransformsSimCLR
from simclr.models import Simclr
from simclr.modules import load_optimizers
from simclr.modules import Loss

from utils.sync_batchnorm import convert_model
from utils import save_model

def train(args, train_loader, model, criterion, optimizer, writer, neg_label, epoch):

    loss_epoch = 0
    for step, (x, _) in enumerate(train_loader):

        x = torch.cat(x, dim=0)
        x = x.cuda(non_blocking=True)

        features = model(x)

        optimizer.zero_grad()
        loss = criterion(features)
        loss.backward()
        optimizer.step()
        args.batch_idx +=1

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(args):



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(args.dataset_dir,
                                                     download=True,
                                                     transform=TransformsSimCLR(size=args.image_size, strength=args.strength))
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(args.dataset_dir,
                                                     download=True,
                                                     transform=TransformsSimCLR(size=args.image_size, strength=args.strength))

    else:
        raise NotImplementedError

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               drop_last=True,
                                               num_workers=args.workers,
                                               sampler=train_sampler,
    )


    model = Simclr(args)
    model = model.to(args.device)
    args.global_step = 0
    optimizer, scheduler = load_optimizers(args, model, len(train_dataset))
    criterion = Loss(args.batch_size, args.temperature)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)

    model = model.to(args.device)
    writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.epochs):

        args.batch_idx = 0
        lr = optimizer.param_groups[0]["lr"]
        neg_label = 0
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer, neg_label, epoch)

        if scheduler:
            scheduler.step()

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {lr}")


    ## end training
    save_model(args, model, filename=os.path.join(writer.log_dir, 'checkpoint.pth.tar'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    main(args)