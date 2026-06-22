import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
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
from common.model_loader import (  # noqa: E402
    create_encoder_feature_extractor,
    get_supported_model_names,
    load_pretrained_model,
    resolve_device,
)
from common.optimizers import create_linear_optimizer  # noqa: E402
from linear_eval import evaluate, parse_pretrain_batch_size, set_random_seed, train_one_epoch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Linear evaluation hyperparameter search.")
    parser.add_argument("--model", choices=get_supported_model_names(), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--dataset-dir", default=str(PROJECT_DIR / "dataset"))
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "lars"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-6, 1e-4])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def create_output_path(args):
    if args.output_path is not None:
        return Path(args.output_path)

    checkpoint_name = Path(args.checkpoint).stem
    return PROJECT_DIR / "results" / "linear_eval_hparam" / f"{args.model}_{checkpoint_name}.json"


def combo_key(combo):
    return (
        combo["optimizer"],
        combo["learning_rate"],
        combo["weight_decay"],
    )


def load_existing_payload(output_path):
    if not output_path.exists():
        return None

    return json.loads(output_path.read_text(encoding="utf-8"))


def create_eval_config(training_config, args):
    eval_config = dict(training_config)
    eval_config["dataset"] = args.dataset
    eval_config["dataset_dir"] = resolve_project_path(PROJECT_DIR, args.dataset_dir)
    eval_config["batch_size"] = args.batch_size
    eval_config["num_workers"] = args.num_workers
    return eval_config


def create_payload(args, checkpoint, combinations):
    return {
        "model": args.model,
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pretrain_batch_size": parse_pretrain_batch_size(args.checkpoint),
        "linear_eval_batch_size": args.batch_size,
        "epochs": args.epochs,
        "momentum": args.momentum,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "finished_at": None,
        "partial": True,
        "planned_count": len(combinations),
        "results": [],
        "best": None,
    }


def summarize_result(combo, history, elapsed_seconds):
    best_epoch = 0
    best_top1 = 0.0
    best_top5 = 0.0

    for epoch_result in history:
        test_metrics = epoch_result["test"]
        if test_metrics["top1"] > best_top1:
            best_epoch = epoch_result["epoch"]
            best_top1 = test_metrics["top1"]
            best_top5 = test_metrics["top5"]

    return {
        "optimizer": combo["optimizer"],
        "learning_rate": combo["learning_rate"],
        "weight_decay": combo["weight_decay"],
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "best_top5": best_top5,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }


def select_best(results):
    if not results:
        return None

    return max(results, key=lambda result: result["best_top1"])


def run_combo(encoder, eval_config, train_loader, test_loader, args, combo, device):
    set_random_seed(args.seed)
    classifier = nn.Linear(eval_config["encoder_feature_dim"], get_num_classes(args.dataset)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_linear_optimizer(
        classifier,
        combo["optimizer"],
        combo["learning_rate"],
        args.momentum,
        combo["weight_decay"],
    )
    history = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(encoder, classifier, train_loader, criterion, optimizer, device, epoch)
        test_metrics = evaluate(encoder, classifier, test_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        })
        print(
            f"combo optimizer={combo['optimizer']} "
            f"lr={combo['learning_rate']} "
            f"wd={combo['weight_decay']} "
            f"epoch={epoch} "
            f"test_top1={test_metrics['top1']:.2f}"
        )

    return summarize_result(combo, history, time.time() - started_at)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    output_path = create_output_path(args)
    combinations = [
        {"optimizer": optimizer_name, "learning_rate": learning_rate, "weight_decay": weight_decay}
        for optimizer_name, learning_rate, weight_decay in itertools.product(
            args.optimizers,
            args.learning_rates,
            args.weight_decays,
        )
    ]

    device = resolve_device(args.device)
    pretrained_model, training_config, checkpoint = load_pretrained_model(args.model, args.checkpoint, device)
    eval_config = create_eval_config(training_config, args)
    encoder = create_encoder_feature_extractor(args.model, pretrained_model).to(device)

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    train_dataset = create_cifar_dataset(
        args.dataset,
        eval_config["dataset_dir"],
        train=True,
        transform=create_linear_train_transform(eval_config),
        download=eval_config["download_dataset"],
    )
    test_dataset = create_cifar_dataset(
        args.dataset,
        eval_config["dataset_dir"],
        train=False,
        transform=create_eval_transform(eval_config),
        download=eval_config["download_dataset"],
    )
    train_loader = create_dataloader(train_dataset, args.batch_size, args.num_workers, True, device)
    test_loader = create_dataloader(test_dataset, args.batch_size, args.num_workers, False, device)

    payload = load_existing_payload(output_path) or create_payload(args, checkpoint, combinations)
    completed_keys = {combo_key(result) for result in payload["results"]}

    for combo in combinations:
        if combo_key(combo) in completed_keys:
            print(f"skip_completed optimizer={combo['optimizer']} lr={combo['learning_rate']} wd={combo['weight_decay']}")
            continue

        result = run_combo(encoder, eval_config, train_loader, test_loader, args, combo, device)
        payload["results"].append(result)
        payload["best"] = select_best(payload["results"])
        write_json(output_path, payload)

    payload["best"] = select_best(payload["results"])
    payload["partial"] = len(payload["results"]) < len(combinations)
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(output_path, payload)
    print(f"result_path={output_path}")
    print(f"best={payload['best']}")


if __name__ == "__main__":
    main()
