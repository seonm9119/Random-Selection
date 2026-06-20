import argparse
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from common.datasets import (  # noqa: E402
    create_cifar_dataset,
    create_dataloader,
    create_eval_transform,
    create_linear_train_transform,
    get_num_classes,
    resolve_project_path,
)
from common.json_utils import write_json  # noqa: E402
from common.metrics import AverageMeter, calculate_topk_accuracy  # noqa: E402
from common.model_loader import (  # noqa: E402
    create_encoder_feature_extractor,
    get_supported_model_names,
    load_pretrained_model,
    resolve_device,
)
from common.optimizers import create_linear_optimizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Linear evaluation for pretrained SSL encoders.")
    parser.add_argument("--model", choices=get_supported_model_names(), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["lars", "sgd", "adam"], default="lars")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_eval_config(training_config, args):
    eval_config = dict(training_config)

    if args.dataset is not None:
        eval_config["dataset"] = args.dataset

    if args.dataset_dir is not None:
        eval_config["dataset_dir"] = args.dataset_dir

    eval_config["dataset_dir"] = resolve_project_path(PROJECT_DIR, eval_config["dataset_dir"])
    eval_config["batch_size"] = args.batch_size
    eval_config["num_workers"] = args.num_workers
    return eval_config


def create_output_path(args, eval_config):
    if args.output_path is not None:
        return args.output_path

    checkpoint_name = Path(args.checkpoint).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{args.model}_{eval_config['dataset']}_{checkpoint_name}_{timestamp}.json"
    return PROJECT_DIR / "results" / "linear_eval" / file_name


def train_one_epoch(encoder, classifier, dataloader, criterion, optimizer, device, epoch):
    classifier.train()
    encoder.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    progress_bar = tqdm(dataloader, desc=f"linear train {epoch}", leave=False)

    for image_batch, label_batch in progress_bar:
        image_batch = image_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        with torch.no_grad():
            feature_batch = encoder(image_batch)

        logits = classifier(feature_batch.detach())
        loss = criterion(logits, label_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        top1, top5 = calculate_topk_accuracy(logits, label_batch, topk=(1, 5))
        batch_size = image_batch.shape[0]
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)
        progress_bar.set_postfix({"loss": f"{loss_meter.average:.4f}", "top1": f"{top1_meter.average:.2f}"})

    return {
        "loss": loss_meter.average,
        "top1": top1_meter.average,
        "top5": top5_meter.average,
    }


@torch.no_grad()
def evaluate(encoder, classifier, dataloader, criterion, device):
    classifier.eval()
    encoder.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for image_batch, label_batch in tqdm(dataloader, desc="linear test", leave=False):
        image_batch = image_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        feature_batch = encoder(image_batch)
        logits = classifier(feature_batch)
        loss = criterion(logits, label_batch)
        top1, top5 = calculate_topk_accuracy(logits, label_batch, topk=(1, 5))
        batch_size = image_batch.shape[0]
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

    return {
        "loss": loss_meter.average,
        "top1": top1_meter.average,
        "top5": top5_meter.average,
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    pretrained_model, training_config, checkpoint = load_pretrained_model(args.model, args.checkpoint, device)
    eval_config = resolve_eval_config(training_config, args)
    encoder = create_encoder_feature_extractor(args.model, pretrained_model).to(device)

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    train_dataset = create_cifar_dataset(
        eval_config["dataset"],
        eval_config["dataset_dir"],
        train=True,
        transform=create_linear_train_transform(eval_config),
        download=eval_config["download_dataset"],
    )
    test_dataset = create_cifar_dataset(
        eval_config["dataset"],
        eval_config["dataset_dir"],
        train=False,
        transform=create_eval_transform(eval_config),
        download=eval_config["download_dataset"],
    )
    train_loader = create_dataloader(train_dataset, args.batch_size, args.num_workers, True, device)
    test_loader = create_dataloader(test_dataset, args.batch_size, args.num_workers, False, device)

    classifier = nn.Linear(eval_config["encoder_feature_dim"], get_num_classes(eval_config["dataset"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_linear_optimizer(classifier, args.optimizer, args.learning_rate, args.momentum, args.weight_decay)
    history = []
    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(encoder, classifier, train_loader, criterion, optimizer, device, epoch)
        test_metrics = evaluate(encoder, classifier, test_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        })

        if test_metrics["top1"] > best_top1:
            best_top1 = test_metrics["top1"]
            best_top5 = test_metrics["top5"]
            best_epoch = epoch

        print(
            f"epoch={epoch} "
            f"train_top1={train_metrics['top1']:.2f} "
            f"test_top1={test_metrics['top1']:.2f} "
            f"test_top5={test_metrics['top5']:.2f}"
        )

    output_payload = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": eval_config["dataset"],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "best_top5": best_top5,
        "history": history,
    }
    output_path = create_output_path(args, eval_config)
    write_json(output_path, output_payload)
    print(f"result_path={output_path}")
    print(f"best_epoch={best_epoch} best_top1={best_top1:.2f} best_top5={best_top5:.2f}")


if __name__ == "__main__":
    main()
