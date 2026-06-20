import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_BIN = PROJECT_DIR / ".venv" / "bin" / "python"
SEARCH_DIR = PROJECT_DIR / "results" / "lr_search" / "simclr"

DEFAULT_LEARNING_RATES = (0.03, 0.075, 0.15, 0.3, 1.0)
DEFAULT_DATASET = "cifar100"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 30
DEFAULT_WARMUP_EPOCHS = 10
DEFAULT_TRAIN_LOSS_STOP_MIN_DELTA = 0.02
DEFAULT_TRAIN_LOSS_STOP_PATIENCE = 5


def get_python_bin():
    if PYTHON_BIN.exists():
        return str(PYTHON_BIN)

    return sys.executable


def parse_learning_rates():
    configured_learning_rates = os.environ.get("RS_LR_SEARCH_VALUES")

    if not configured_learning_rates:
        return DEFAULT_LEARNING_RATES

    return tuple(float(learning_rate) for learning_rate in configured_learning_rates.split(","))


def create_run_name(learning_rate):
    dataset = os.environ.get("RS_LR_SEARCH_DATASET", DEFAULT_DATASET)
    batch_size = int(os.environ.get("RS_LR_SEARCH_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    learning_rate_text = f"{learning_rate:g}".replace(".", "p")
    return f"{dataset}_batch_{batch_size}_lr_{learning_rate_text}"


def create_environment(learning_rate, run_name):
    dataset = os.environ.get("RS_LR_SEARCH_DATASET", DEFAULT_DATASET)
    batch_size = int(os.environ.get("RS_LR_SEARCH_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    epochs = int(os.environ.get("RS_LR_SEARCH_EPOCHS", DEFAULT_EPOCHS))
    warmup_epochs = int(os.environ.get("RS_LR_SEARCH_WARMUP_EPOCHS", DEFAULT_WARMUP_EPOCHS))
    stop_start_epoch = int(os.environ.get("RS_LR_SEARCH_STOP_START_EPOCH", warmup_epochs + 1))
    output_dir = SEARCH_DIR / run_name

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONWARNINGS": "ignore::FutureWarning",
            "RS_DATASET": dataset,
            "RS_BATCH_SIZE": str(batch_size),
            "RS_EPOCHS": str(epochs),
            "RS_LEARNING_RATE": str(learning_rate),
            "RS_WARMUP_EPOCHS": str(warmup_epochs),
            "RS_OUTPUT_DIR": str(output_dir),
            "RS_CONFIG_FILE_NAME": "config.json",
            "RS_EARLY_STOP_ENABLED": "false",
            "RS_TRAIN_LOSS_STOP_ENABLED": "true",
            "RS_TRAIN_LOSS_STOP_START_EPOCH": str(stop_start_epoch),
            "RS_TRAIN_LOSS_STOP_MIN_DELTA": os.environ.get(
                "RS_LR_SEARCH_STOP_MIN_DELTA",
                str(DEFAULT_TRAIN_LOSS_STOP_MIN_DELTA),
            ),
            "RS_TRAIN_LOSS_STOP_PATIENCE": os.environ.get(
                "RS_LR_SEARCH_STOP_PATIENCE",
                str(DEFAULT_TRAIN_LOSS_STOP_PATIENCE),
            ),
            "RS_SAVE_BEST_CHECKPOINT": "false",
            "RS_SUPPRESS_EXTERNAL_PROGRESS": "true",
        }
    )
    return environment


def run_learning_rate_candidate(learning_rate):
    run_name = create_run_name(learning_rate)
    run_dir = SEARCH_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    environment = create_environment(learning_rate, run_name)
    command = [get_python_bin(), "benchmark/simclr/train.py"]

    with log_path.open("w", encoding="utf-8") as log_file:
        completed_process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    summary = summarize_candidate(run_dir, learning_rate)
    summary["return_code"] = completed_process.returncode
    summary["console_log"] = str(log_path.relative_to(PROJECT_DIR))
    return summary


def summarize_candidate(run_dir, learning_rate):
    console_log_path = run_dir / "console.log"
    rows = read_console_epoch_rows(console_log_path)

    if not rows:
        return {
            "learning_rate": learning_rate,
            "epochs": 0,
            "status": "no_epoch_log",
        }

    training_config = read_training_config(run_dir)
    train_loss_stop_start_epoch = training_config.get("train_loss_stop_start_epoch", DEFAULT_WARMUP_EPOCHS + 1)
    first_row = rows[0]
    last_row = rows[-1]
    active_rows = [row for row in rows if row["epoch"] >= train_loss_stop_start_epoch]
    first_active_row = active_rows[0] if active_rows else first_row
    best_train_row = min(rows, key=lambda row: row["train_loss"])
    best_val_row = min(rows, key=lambda row: row["val_loss"])
    stopped_by_train_loss = console_log_contains(console_log_path, "train_loss_stop ")

    return {
        "learning_rate": learning_rate,
        "epochs": len(rows),
        "first_train_loss": first_row["train_loss"],
        "first_active_epoch": first_active_row["epoch"],
        "first_active_train_loss": first_active_row["train_loss"],
        "last_epoch": last_row["epoch"],
        "last_train_loss": last_row["train_loss"],
        "last_val_loss": last_row["val_loss"],
        "best_train_epoch": best_train_row["epoch"],
        "best_train_loss": best_train_row["train_loss"],
        "best_val_epoch": best_val_row["epoch"],
        "best_val_loss": best_val_row["val_loss"],
        "post_warmup_train_loss_drop": first_active_row["train_loss"] - best_train_row["train_loss"],
        "stopped_by_train_loss": stopped_by_train_loss,
        "status": "completed",
    }


def read_console_epoch_rows(console_log_path):
    if not console_log_path.exists():
        return []

    rows = []
    epoch_pattern = re.compile(
        r"^epoch=(?P<epoch>\d+) "
        r"train_loss=(?P<train_loss>[-+0-9.eE]+) "
        r"val_loss=(?P<val_loss>[-+0-9.eE]+) "
        r"lr=(?P<learning_rate>[-+0-9.eE]+) "
        r"best_val_loss=(?P<best_val_loss>[-+0-9.eE]+) "
        r"early_stop_wait=(?P<early_stop_wait>\d+)"
    )

    for line in console_log_path.read_text(encoding="utf-8").splitlines():
        epoch_match = epoch_pattern.match(line.strip())

        if not epoch_match:
            continue

        rows.append(
            {
                "epoch": int(epoch_match.group("epoch")),
                "train_loss": float(epoch_match.group("train_loss")),
                "val_loss": float(epoch_match.group("val_loss")),
                "learning_rate": float(epoch_match.group("learning_rate")),
                "best_val_loss": float(epoch_match.group("best_val_loss")),
                "early_stop_wait": int(epoch_match.group("early_stop_wait")),
            }
        )

    return rows


def read_training_config(run_dir):
    config_paths = sorted(run_dir.glob("*.json"))

    if not config_paths:
        return {}

    return json.loads(config_paths[0].read_text(encoding="utf-8"))


def console_log_contains(console_log_path, text):
    if not console_log_path.exists():
        return False

    return text in console_log_path.read_text(encoding="utf-8")


def write_summary(summaries):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SEARCH_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def main():
    summaries = []

    for learning_rate in parse_learning_rates():
        print(f"start lr={learning_rate}", flush=True)
        summary = run_learning_rate_candidate(learning_rate)
        summaries.append(summary)
        print(
            "done "
            f"lr={learning_rate} epochs={summary.get('epochs')} "
            f"last_train_loss={summary.get('last_train_loss')} "
            f"drop={summary.get('post_warmup_train_loss_drop')}",
            flush=True,
        )

    summary_path = write_summary(summaries)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
