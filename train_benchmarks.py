import os
import subprocess
import sys
from pathlib import Path

from common.training_outputs import create_dataset_batch_best_basename


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_BIN = PROJECT_DIR / ".venv" / "bin" / "python"

FINAL_PRETRAIN_EPOCHS = int(os.environ.get("RS_BENCHMARK_EPOCHS", "400"))
FINAL_PRETRAIN_SAVE_EVERY = int(os.environ.get("RS_BENCHMARK_SAVE_EVERY", "100"))
FINAL_PRETRAIN_EARLY_STOP_ENABLED = os.environ.get("RS_BENCHMARK_EARLY_STOP_ENABLED", "false")
FINAL_PRETRAIN_EARLY_STOP_MIN_DELTA = os.environ.get("RS_BENCHMARK_EARLY_STOP_MIN_DELTA", "0.0003")
FINAL_PRETRAIN_EARLY_STOP_PATIENCE = os.environ.get("RS_BENCHMARK_EARLY_STOP_PATIENCE", "5")
FINAL_PRETRAIN_TRAIN_LOSS_STOP_ENABLED = os.environ.get("RS_BENCHMARK_TRAIN_LOSS_STOP_ENABLED", "true")
FINAL_PRETRAIN_TRAIN_LOSS_STOP_START_EPOCH = os.environ.get("RS_BENCHMARK_TRAIN_LOSS_STOP_START_EPOCH", "0")
FINAL_PRETRAIN_TRAIN_LOSS_STOP_MIN_DELTA = os.environ.get("RS_BENCHMARK_TRAIN_LOSS_STOP_MIN_DELTA", "0.01")
FINAL_PRETRAIN_TRAIN_LOSS_STOP_PATIENCE = os.environ.get("RS_BENCHMARK_TRAIN_LOSS_STOP_PATIENCE", "5")
MODEL_FILTER = os.environ.get("RS_BENCHMARK_MODELS", "")
DATASET_FILTER = os.environ.get("RS_BENCHMARK_DATASETS", "")
BATCH_SIZE_FILTER = os.environ.get("RS_BENCHMARK_BATCH_SIZES", "")

DATASETS = ("cifar100", "cifar10")

BASE_EXPERIMENTS = (
    {
        "model": "simclr",
        "script": "benchmark/simclr/train.py",
        "batch_size": 1024,
        "output_dir": "benchmark/simclr/pretrained",
    },
    {
        "model": "simclr",
        "script": "benchmark/simclr/train.py",
        "batch_size": 512,
        "output_dir": "benchmark/simclr/pretrained",
    },
    {
        "model": "simclr",
        "script": "benchmark/simclr/train.py",
        "batch_size": 256,
        "output_dir": "benchmark/simclr/pretrained",
    },
    {
        "model": "simclr",
        "script": "benchmark/simclr/train.py",
        "batch_size": 128,
        "output_dir": "benchmark/simclr/pretrained",
    },
    {
        "model": "byol",
        "script": "benchmark/byol/train.py",
        "batch_size": 1024,
        "output_dir": "benchmark/byol/pretrained",
    },
    {
        "model": "simsiam",
        "script": "benchmark/simsiam/train.py",
        "batch_size": 1024,
        "output_dir": "benchmark/simsiam/pretrained",
    },
)


def get_python_bin():
    if PYTHON_BIN.exists():
        return str(PYTHON_BIN)

    return sys.executable


def create_experiment_env(experiment):
    environment = os.environ.copy()
    artifact_basename = create_experiment_artifact_basename(experiment)
    environment_updates = {
        "PYTHONWARNINGS": "ignore::FutureWarning",
        "RS_DATASET": experiment["dataset"],
        "RS_EPOCHS": str(FINAL_PRETRAIN_EPOCHS),
        "RS_BATCH_SIZE": str(experiment["batch_size"]),
        "RS_SAVE_EVERY": str(FINAL_PRETRAIN_SAVE_EVERY),
        "RS_OUTPUT_DIR": experiment["output_dir"],
        "RS_EARLY_STOP_ENABLED": FINAL_PRETRAIN_EARLY_STOP_ENABLED,
        "RS_EARLY_STOP_MIN_DELTA": FINAL_PRETRAIN_EARLY_STOP_MIN_DELTA,
        "RS_EARLY_STOP_PATIENCE": FINAL_PRETRAIN_EARLY_STOP_PATIENCE,
        "RS_TRAIN_LOSS_STOP_ENABLED": FINAL_PRETRAIN_TRAIN_LOSS_STOP_ENABLED,
        "RS_TRAIN_LOSS_STOP_START_EPOCH": FINAL_PRETRAIN_TRAIN_LOSS_STOP_START_EPOCH,
        "RS_TRAIN_LOSS_STOP_MIN_DELTA": FINAL_PRETRAIN_TRAIN_LOSS_STOP_MIN_DELTA,
        "RS_TRAIN_LOSS_STOP_PATIENCE": FINAL_PRETRAIN_TRAIN_LOSS_STOP_PATIENCE,
        "RS_SUPPRESS_EXTERNAL_PROGRESS": "true",
        "RS_CONFIG_FILE_NAME": f"{artifact_basename}.json",
        "RS_CHECKPOINT_FILE_TEMPLATE": f"{artifact_basename}.pt",
        "RS_BEST_CHECKPOINT_FILE_NAME": f"{artifact_basename}.pt",
    }

    environment.update(environment_updates)
    return environment


def create_experiment_artifact_basename(experiment):
    return create_dataset_batch_best_basename(experiment["dataset"], experiment["batch_size"])


def create_log_path(experiment):
    log_dir = PROJECT_DIR / experiment["output_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{create_experiment_artifact_basename(experiment)}.log"


def parse_filter_values(filter_text):
    if not filter_text:
        return set()

    return {value.strip() for value in filter_text.split(",") if value.strip()}


def should_run_experiment(experiment):
    model_filter = parse_filter_values(MODEL_FILTER)
    dataset_filter = parse_filter_values(DATASET_FILTER)
    batch_size_filter = parse_filter_values(BATCH_SIZE_FILTER)

    if model_filter and experiment["model"] not in model_filter:
        return False

    if dataset_filter and experiment["dataset"] not in dataset_filter:
        return False

    if batch_size_filter and str(experiment["batch_size"]) not in batch_size_filter:
        return False

    return True


def run_experiment(experiment):
    log_path = create_log_path(experiment)
    command = [get_python_bin(), experiment["script"]]

    print(f"start {experiment['name']} log={log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        completed_process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=create_experiment_env(experiment),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if completed_process.returncode != 0:
        raise RuntimeError(f"{experiment['name']} failed. Check {log_path}")

    print(f"done {experiment['name']}", flush=True)


def main():
    selected_count = 0

    for dataset_name in DATASETS:
        for base_experiment in BASE_EXPERIMENTS:
            experiment = dict(base_experiment)
            experiment["dataset"] = dataset_name
            experiment["name"] = f"{dataset_name}_{experiment['model']}_batch_{experiment['batch_size']}"
            if not should_run_experiment(experiment):
                continue

            selected_count += 1
            run_experiment(experiment)

    if selected_count == 0:
        raise RuntimeError("No benchmark experiments selected.")


if __name__ == "__main__":
    main()
