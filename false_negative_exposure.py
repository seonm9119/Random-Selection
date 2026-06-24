import argparse
from pathlib import Path
import sys
import time

import torch
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from common.datasets import create_cifar_dataset, create_dataloader, create_eval_transform, resolve_project_path  # noqa: E402
from common.false_negative import calculate_false_negative_stats  # noqa: E402
from common.json_utils import write_json  # noqa: E402
from common.model_loader import load_default_training_config, resolve_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Measure label-based false negative exposure for CIFAR batches.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--rscl-k", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def create_output_path(args):
    if args.output_path is not None:
        return args.output_path

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{args.dataset}_false_negative_exposure_{timestamp}.json"
    return PROJECT_DIR / "output" / "analysis" / file_name


def average_batch_stats(batch_stats):
    summary = {
        "candidate_count": batch_stats[0]["candidate_count"],
        "simclr_false_negative_count": 0.0,
        "simclr_false_negative_ratio": 0.0,
        "rscl": {},
    }

    for current_stats in batch_stats:
        summary["simclr_false_negative_count"] += current_stats["simclr_false_negative_count"]
        summary["simclr_false_negative_ratio"] += current_stats["simclr_false_negative_ratio"]

        for selected_negative_count, rscl_stats in current_stats["rscl"].items():
            if selected_negative_count not in summary["rscl"]:
                summary["rscl"][selected_negative_count] = {
                    "effective_selected_negative_count": rscl_stats["effective_selected_negative_count"],
                    "expected_false_negative_count": 0.0,
                    "expected_false_negative_ratio": 0.0,
                    "false_negative_selection_probability": 0.0,
                    "exposure_fraction_vs_simclr": rscl_stats["exposure_fraction_vs_simclr"],
                }

            summary["rscl"][selected_negative_count]["expected_false_negative_count"] += rscl_stats[
                "expected_false_negative_count"
            ]
            summary["rscl"][selected_negative_count]["expected_false_negative_ratio"] += rscl_stats[
                "expected_false_negative_ratio"
            ]
            summary["rscl"][selected_negative_count]["false_negative_selection_probability"] += rscl_stats[
                "false_negative_selection_probability"
            ]

    batch_count = len(batch_stats)
    summary["simclr_false_negative_count"] /= batch_count
    summary["simclr_false_negative_ratio"] /= batch_count

    for rscl_stats in summary["rscl"].values():
        rscl_stats["expected_false_negative_count"] /= batch_count
        rscl_stats["expected_false_negative_ratio"] /= batch_count
        rscl_stats["false_negative_selection_probability"] /= batch_count

    return summary


def measure_batch_size(dataset, batch_size, args, device):
    dataloader = create_dataloader(dataset, batch_size, args.num_workers, True, device, drop_last=True)
    batch_stats = []

    for batch_index, (_, label_batch) in enumerate(tqdm(dataloader, desc=f"batch {batch_size}", leave=False), start=1):
        label_batch = label_batch.to(device)
        batch_stats.append(calculate_false_negative_stats(label_batch, args.rscl_k))

        if batch_index >= args.num_batches:
            break

    if not batch_stats:
        raise RuntimeError("No batches were collected for false negative exposure analysis.")

    return average_batch_stats(batch_stats)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    training_config = load_default_training_config("rscl")
    training_config["dataset"] = args.dataset
    training_config["dataset_dir"] = resolve_project_path(PROJECT_DIR, args.dataset_dir)
    transform = create_eval_transform(training_config)
    dataset = create_cifar_dataset(args.dataset, training_config["dataset_dir"], True, transform, True)
    results = []

    for batch_size in args.batch_sizes:
        summary = measure_batch_size(dataset, batch_size, args, device)
        summary["batch_size"] = batch_size
        results.append(summary)

    output_payload = {
        "dataset": args.dataset,
        "dataset_dir": training_config["dataset_dir"],
        "num_batches": args.num_batches,
        "rscl_k": args.rscl_k,
        "results": results,
    }
    output_path = create_output_path(args)
    write_json(output_path, output_payload)
    print(f"result_path={output_path}")

    for summary in results:
        print(
            f"batch_size={summary['batch_size']} "
            f"simclr_fn={summary['simclr_false_negative_count']:.4f} "
            f"simclr_ratio={summary['simclr_false_negative_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
