import argparse
import json
import random
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset


PROJECT_DIR = Path(__file__).resolve().parents[1]
RSCL_DIR = PROJECT_DIR / "rscl"
sys.path.insert(0, str(RSCL_DIR))
sys.path.insert(0, str(PROJECT_DIR))
warnings.filterwarnings("ignore", category=FutureWarning)

import config  # noqa: E402
import train  # noqa: E402
from loss import RandomSelectionContrastiveLoss  # noqa: E402
from model import RsclModel  # noqa: E402
from optimizer import create_learning_rate_schedule, create_optimizer, set_optimizer_learning_rate  # noqa: E402


DEFAULT_BATCH_SIZES = (1024, 512, 256)
DEFAULT_LEARNING_RATES = (0.5, 1.0, 1.5)
DEFAULT_TEMPERATURES = (0.1, 0.5, 1.0)
DEFAULT_NEGATIVE_COUNTS = ()


def parse_args():
    default_config = config.get_training_config()
    parser = argparse.ArgumentParser(description="Smoke search RSCL hyperparameter combinations.")
    parser.add_argument("--dataset", choices=default_config["supported_datasets"], default=default_config["cifar10_dataset_name"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=list(DEFAULT_LEARNING_RATES))
    parser.add_argument("--temperatures", nargs="+", type=float, default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--negative-counts", nargs="+", type=int, default=list(DEFAULT_NEGATIVE_COUNTS))
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=default_config["warmup_epochs"])
    parser.add_argument("--observe-epochs", type=int, default=50)
    parser.add_argument("--val-every-epochs", type=int, default=5)
    parser.add_argument("--val-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=default_config["seed"])
    parser.add_argument("--device", default=default_config["device"])
    parser.add_argument("--amp", dest="amp", action="store_true", default=default_config["amp"])
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--wait-for-gpu-memory", action="store_true")
    parser.add_argument("--gpu-memory-poll-seconds", type=int, default=60)
    parser.add_argument("--output-path", default="scripts/results/smoke/rscl_hparam_search.json")
    args = parser.parse_args()

    if args.warmup_epochs < 1:
        parser.error("--warmup-epochs must be at least 1.")
    if args.observe_epochs < 1:
        parser.error("--observe-epochs must be at least 1.")
    if args.val_every_epochs < 1:
        parser.error("--val-every-epochs must be at least 1.")
    if any(negative_count < 1 for negative_count in args.negative_counts):
        parser.error("--negative-counts must contain only positive integers.")

    return args


def get_required_free_memory_mb(batch_size):
    required_memory_by_batch_size = {
        1024: 30000,
        512: 16000,
        256: 9000,
    }
    return required_memory_by_batch_size.get(batch_size, 30000)


def wait_for_gpu_memory(args, device, batch_size):
    if not args.wait_for_gpu_memory or device.type != "cuda":
        return

    required_free_memory_mb = get_required_free_memory_mb(batch_size)

    while True:
        free_memory_bytes, _ = torch.cuda.mem_get_info(device)
        free_memory_mb = free_memory_bytes / 1024 ** 2

        if free_memory_mb >= required_free_memory_mb:
            return

        print(
            f"wait_gpu_memory batch={batch_size} "
            f"free={free_memory_mb:.1f}MB required={required_free_memory_mb}MB",
            flush=True,
        )
        time.sleep(args.gpu_memory_poll_seconds)


def resolve_project_path(path_text):
    configured_path = Path(path_text)

    if configured_path.is_absolute():
        return str(configured_path)

    return str(PROJECT_DIR / configured_path)


def create_search_training_config(args, batch_size, learning_rate, temperature, negative_count=None):
    training_config = config.get_training_config()
    training_config["dataset"] = args.dataset
    training_config["epochs"] = args.warmup_epochs + args.observe_epochs
    training_config["batch_size"] = batch_size
    training_config["learning_rate"] = learning_rate
    training_config["temperature"] = temperature
    training_config["warmup_epochs"] = args.warmup_epochs
    training_config["num_workers"] = args.num_workers
    training_config["seed"] = args.seed
    training_config["device"] = args.device
    training_config["amp"] = args.amp
    training_config["max_batch_size"] = max(training_config["max_batch_size"], batch_size)
    training_config["validation_size"] = 0
    training_config["early_stop_enabled"] = False
    training_config["train_loss_stop_enabled"] = False
    training_config["dataloader_shuffle"] = False
    training_config["dataloader_persistent_workers"] = training_config["num_workers"] > 0
    training_config["suppress_external_progress"] = True

    if training_config["device"] == training_config["auto_device"]:
        training_config["device"] = training_config["cuda_device"] if torch.cuda.is_available() else training_config["cpu_device"]

    training_config["dataset_dir"] = resolve_project_path(training_config["dataset_dir"])
    training_config["output_dir"] = resolve_project_path(training_config["output_dir"])
    training_config["encoder_feature_dim"] = training_config["backbone_feature_dims"][training_config["backbone_name"]]
    training_config["projection_hidden_dim"] = training_config["projection_hidden_dims"][training_config["backbone_name"]]
    training_config["amp"] = training_config["amp"] and training_config["device"] == training_config["cuda_device"]
    train.resolve_fixed_random_negative_count(training_config)
    return training_config


def create_smoke_dataloaders(training_config, steps_per_epoch, val_steps):
    pair_transform = train.RsclPairTransform(training_config)
    dataset = train.create_cifar_dataset(training_config, pair_transform)
    train_sample_count = min(len(dataset), training_config["batch_size"] * steps_per_epoch)
    val_sample_count = min(len(dataset) - train_sample_count, training_config["batch_size"] * val_steps)
    train_dataset = Subset(dataset, range(train_sample_count))
    val_dataset = Subset(dataset, range(train_sample_count, train_sample_count + val_sample_count))
    device = torch.device(training_config["device"])
    train_dataloader = train.create_dataloader(train_dataset, training_config, device, shuffle=False)
    val_config = {
        **training_config,
        "num_workers": 0,
        "dataloader_persistent_workers": False,
    }
    val_dataloader = train.create_dataloader(val_dataset, val_config, device, shuffle=False)
    return train_dataloader, val_dataloader


def train_smoke_epoch(model, criterion, dataloader, optimizer, learning_rate_schedule, scaler, device, training_config, epoch):
    losses = []
    model.train()

    for step_index, ((first_view_batch, second_view_batch), _) in enumerate(
        dataloader,
        start=training_config["step_start_index"],
    ):
        first_view_batch = first_view_batch.to(device, non_blocking=True)
        second_view_batch = second_view_batch.to(device, non_blocking=True)
        image_batch = torch.cat([first_view_batch, second_view_batch], dim=training_config["batch_concat_dim"])
        global_step = (epoch - training_config["training_start_epoch"]) * len(dataloader)
        global_step += step_index - training_config["step_start_index"]
        learning_rate = learning_rate_schedule.get_learning_rate(global_step)
        set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=training_config["optimizer_set_to_none"])

        with torch.cuda.amp.autocast(enabled=training_config["amp"]):
            _, projection_batch = model(image_batch)
            training_loss = criterion(projection_batch)

        if not torch.isfinite(training_loss):
            raise RuntimeError(f"non-finite train loss: {training_loss.item()}")

        scaler.scale(training_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(training_loss.item())

    return losses


def evaluate_smoke_loss(model, criterion, dataloader, device, training_config, seed):
    losses = []
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()
    cuda_random_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None

    train.set_random_seed({**training_config, "seed": seed})
    model.eval()

    try:
        with torch.no_grad():
            for (first_view_batch, second_view_batch), _ in dataloader:
                first_view_batch = first_view_batch.to(device, non_blocking=True)
                second_view_batch = second_view_batch.to(device, non_blocking=True)
                image_batch = torch.cat([first_view_batch, second_view_batch], dim=training_config["batch_concat_dim"])

                with torch.cuda.amp.autocast(enabled=training_config["amp"]):
                    _, projection_batch = model(image_batch)
                    validation_loss = criterion(projection_batch)

                if not torch.isfinite(validation_loss):
                    raise RuntimeError(f"non-finite validation loss: {validation_loss.item()}")

                losses.append(validation_loss.item())
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
        torch.set_rng_state(torch_random_state)

        if cuda_random_states is not None:
            torch.cuda.set_rng_state_all(cuda_random_states)

    return sum(losses) / len(losses)


def summarize_losses(losses):
    first_loss = losses[0]
    last_loss = losses[-1]
    loss_delta = first_loss - last_loss

    return {
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "mean_train_loss": sum(losses) / len(losses),
        "train_loss_delta": loss_delta,
        "train_loss_drop_ratio": loss_delta / max(abs(first_loss), 1e-12),
    }


def summarize_validation_losses(val_measurements):
    val_epochs = [measurement["epoch"] for measurement in val_measurements]
    val_losses = [measurement["loss"] for measurement in val_measurements]
    post_warmup_val_loss = val_losses[0]
    final_val_loss = val_losses[-1]
    val_loss_delta = post_warmup_val_loss - final_val_loss
    val_loss_changes = [
        current_loss - previous_loss
        for previous_loss, current_loss in zip(val_losses, val_losses[1:])
    ]
    val_loss_increases = [
        loss_change
        for loss_change in val_loss_changes
        if loss_change > 1e-8
    ]
    max_val_loss_increase = max(val_loss_increases, default=0.0)
    max_val_loss_increase_ratio = max_val_loss_increase / max(abs(post_warmup_val_loss), 1e-12)
    val_loss_drop_ratio = val_loss_delta / max(abs(post_warmup_val_loss), 1e-12)
    stability_penalty = max_val_loss_increase_ratio + 0.02 * len(val_loss_increases)

    return {
        "post_warmup_epoch": val_epochs[0],
        "final_observed_epoch": val_epochs[-1],
        "post_warmup_val_loss": post_warmup_val_loss,
        "final_val_loss": final_val_loss,
        "min_observed_val_loss": min(val_losses),
        "val_epochs": val_epochs,
        "val_losses": val_losses,
        "val_loss_delta": val_loss_delta,
        "val_loss_drop_ratio": val_loss_drop_ratio,
        "val_loss_increase_count": len(val_loss_increases),
        "max_val_loss_increase": max_val_loss_increase,
        "max_val_loss_increase_ratio": max_val_loss_increase_ratio,
        "val_loss_monotonic_after_warmup": len(val_loss_increases) == 0,
        "val_loss_improved_after_warmup": final_val_loss < post_warmup_val_loss,
        "selection_score": val_loss_drop_ratio - stability_penalty,
    }


def should_evaluate_validation(epoch, last_warmup_epoch, final_epoch, val_every_epochs):
    if epoch == last_warmup_epoch:
        return True

    if epoch == final_epoch:
        return True

    return epoch > last_warmup_epoch and (epoch - last_warmup_epoch) % val_every_epochs == 0


def get_peak_memory(device):
    if device.type != "cuda":
        return {
            "peak_memory_mb": 0.0,
            "reserved_memory_mb": 0.0,
        }

    torch.cuda.synchronize(device)
    return {
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "reserved_memory_mb": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    }


def run_combination(args, batch_size, learning_rate, temperature, negative_count):
    training_config = create_search_training_config(args, batch_size, learning_rate, temperature, negative_count)
    train.set_random_seed(training_config)
    device = torch.device(training_config["device"])

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.time()

    try:
        wait_for_gpu_memory(args, device, batch_size)
        train_dataloader, val_dataloader = create_smoke_dataloaders(
            training_config,
            args.steps_per_epoch,
            args.val_steps,
        )
        train.update_training_step_config(training_config, train_dataloader)
        model = RsclModel(training_config).to(device)
        criterion = RandomSelectionContrastiveLoss(training_config).to(device)
        optimizer = create_optimizer(model, training_config)
        learning_rate_schedule = create_learning_rate_schedule(training_config)
        scaler = torch.cuda.amp.GradScaler(enabled=training_config["amp"])
        train_losses = []
        val_measurements = []

        last_warmup_epoch = training_config["warmup_epochs"]
        last_epoch = training_config["epochs"] + training_config["training_start_epoch"]
        final_epoch = last_epoch - 1

        for epoch in range(training_config["training_start_epoch"], last_epoch):
            train_losses.extend(
                train_smoke_epoch(
                    model,
                    criterion,
                    train_dataloader,
                    optimizer,
                    learning_rate_schedule,
                    scaler,
                    device,
                    training_config,
                    epoch,
                )
            )

            if should_evaluate_validation(epoch, last_warmup_epoch, final_epoch, args.val_every_epochs):
                val_loss = evaluate_smoke_loss(
                    model,
                    criterion,
                    val_dataloader,
                    device,
                    training_config,
                    args.seed + 1000,
                )
                val_measurements.append({
                    "epoch": epoch,
                    "loss": val_loss,
                })

        train_summary = summarize_losses(train_losses)
        validation_summary = summarize_validation_losses(val_measurements)
        memory_summary = get_peak_memory(device)

        return {
            "ok": True,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "scaled_learning_rate": training_config["scaled_learning_rate"],
            "temperature": temperature,
            "negative_count": negative_count,
            "warmup_epochs": training_config["warmup_epochs"],
            "observe_epochs": args.observe_epochs,
            "val_every_epochs": args.val_every_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "train_steps": len(train_losses),
            "val_steps": args.val_steps,
            "score": validation_summary["selection_score"],
            "elapsed_seconds": time.time() - started_at,
            **train_summary,
            **validation_summary,
            **memory_summary,
        }
    except RuntimeError as error:
        error_text = str(error)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "ok": False,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "negative_count": negative_count,
            "error_type": "cuda_oom" if "out of memory" in error_text.lower() else "runtime_error",
            "error": error_text.split("\n")[0],
            "elapsed_seconds": time.time() - started_at,
        }


def get_best_result(batch_results):
    successful_results = [result for result in batch_results if result["ok"]]

    if not successful_results:
        return None

    return max(
        successful_results,
        key=lambda result: (
            result["val_loss_improved_after_warmup"],
            -result["val_loss_increase_count"],
            -result["max_val_loss_increase_ratio"],
            result["val_loss_drop_ratio"],
            -result["final_val_loss"],
        ),
    )


def create_results_payload(args, results, best_by_batch, partial):
    return {
        "note": "RSCL smoke search uses SimCLR learning-rate and temperature sweeps plus random negative count, ranking validation loss improvement after warmup with stability penalties.",
        "partial": partial,
        "dataset": args.dataset,
        "steps_per_epoch": args.steps_per_epoch,
        "warmup_epochs": args.warmup_epochs,
        "observe_epochs": args.observe_epochs,
        "val_every_epochs": args.val_every_epochs,
        "val_steps": args.val_steps,
        "batch_sizes": args.batch_sizes,
        "learning_rates": args.learning_rates,
        "temperatures": args.temperatures,
        "negative_counts": args.negative_counts,
        "best_by_batch": best_by_batch,
        "results": results,
    }


def write_results(output_path, results):
    resolved_output_path = PROJECT_DIR / output_path
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return resolved_output_path


def load_existing_results(output_path):
    resolved_output_path = PROJECT_DIR / output_path

    if not resolved_output_path.exists():
        return []

    payload = json.loads(resolved_output_path.read_text(encoding="utf-8"))
    return payload.get("results", [])


def create_result_key(result):
    return (
        result["batch_size"],
        result["learning_rate"],
        result["temperature"],
        result["negative_count"],
    )


def build_best_by_batch(results):
    best_by_batch = {}
    batch_sizes = sorted({result["batch_size"] for result in results})

    for batch_size in batch_sizes:
        batch_results = [result for result in results if result["batch_size"] == batch_size]
        best_by_batch[str(batch_size)] = get_best_result(batch_results)

    return best_by_batch


def run_search(args):
    results = load_existing_results(args.output_path)
    result_keys = {create_result_key(result) for result in results}
    best_by_batch = build_best_by_batch(results)

    for batch_size in args.batch_sizes:
        fixed_negative_count = batch_size // config.RANDOM_NEGATIVE_COUNT_BATCH_SIZE_DIVISOR
        for learning_rate, temperature in product(args.learning_rates, args.temperatures):
            negative_count = fixed_negative_count
            result_key = (batch_size, learning_rate, temperature, negative_count)
            if result_key in result_keys:
                print(
                    f"skip batch={batch_size} lr={learning_rate} temp={temperature} neg={negative_count}",
                    flush=True,
                )
                continue

            result = run_combination(args, batch_size, learning_rate, temperature, negative_count)
            results.append(result)
            result_keys.add(result_key)

            if result["ok"]:
                print(
                    "batch={batch_size} lr={learning_rate} temp={temperature} neg={negative_count} "
                    "score={score:.4f} drop={val_loss_drop_ratio:.4f} "
                    "bumps={val_loss_increase_count} val={post_warmup_val_loss:.4f}->{final_val_loss:.4f} "
                    "epochs={post_warmup_epoch}->{final_observed_epoch} "
                    "train={first_train_loss:.4f}->{last_train_loss:.4f} peak={peak_memory_mb:.1f}MB".format(
                        **result
                    ),
                    flush=True,
                )
            else:
                print(
                    f"batch={batch_size} lr={learning_rate} temp={temperature} neg={negative_count} "
                    f"fail {result['error_type']}: {result['error']}",
                    flush=True,
                )

            best_by_batch = build_best_by_batch(results)
            write_results(args.output_path, create_results_payload(args, results, best_by_batch, partial=True))

    return create_results_payload(args, results, best_by_batch, partial=False)


def main():
    args = parse_args()
    results = run_search(args)
    output_path = write_results(args.output_path, results)
    print(f"result_path={output_path}", flush=True)


if __name__ == "__main__":
    main()
