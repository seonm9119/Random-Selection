import contextlib
import os
import sys

import torch
from torch.utils.data import random_split


TRUE_TEXT_VALUES = ("1", "true", "yes", "y", "on")
FALSE_TEXT_VALUES = ("0", "false", "no", "n", "off")


def parse_bool(value):
    lowered_value = value.strip().lower()

    if lowered_value in TRUE_TEXT_VALUES:
        return True

    if lowered_value in FALSE_TEXT_VALUES:
        return False

    raise ValueError(f"Cannot parse boolean value: {value}")


def apply_environment_overrides(training_config):
    override_specs = {
        "RS_DATASET": ("dataset", str),
        "RS_OUTPUT_DIR": ("output_dir", str),
        "RS_CONFIG_FILE_NAME": ("config_file_name", str),
        "RS_TRAIN_LOG_FILE_NAME": ("train_log_file_name", str),
        "RS_CHECKPOINT_FILE_TEMPLATE": ("checkpoint_file_template", str),
        "RS_BEST_CHECKPOINT_FILE_NAME": ("best_checkpoint_file_name", str),
        "RS_EPOCHS": ("epochs", int),
        "RS_BATCH_SIZE": ("batch_size", int),
        "RS_MAX_BATCH_SIZE": ("max_batch_size", int),
        "RS_BACKBONE_NAME": ("backbone_name", str),
        "RS_TEMPERATURE": ("temperature", float),
        "RS_LEARNING_RATE": ("learning_rate", float),
        "RS_LEARNING_RATE_SCALING": ("learning_rate_scaling", str),
        "RS_WARMUP_EPOCHS": ("warmup_epochs", int),
        "RS_WEIGHT_DECAY": ("weight_decay", float),
        "RS_NUM_WORKERS": ("num_workers", int),
        "RS_DEVICE": ("device", str),
        "RS_AMP": ("amp", parse_bool),
        "RS_SAVE_EVERY": ("save_every", int),
        "RS_SAVE_BEST_CHECKPOINT": ("save_best_checkpoint", parse_bool),
        "RS_VALIDATION_SIZE": ("validation_size", int),
        "RS_VALIDATION_SPLIT_SEED": ("validation_split_seed", int),
        "RS_EARLY_STOP_ENABLED": ("early_stop_enabled", parse_bool),
        "RS_EARLY_STOP_MIN_DELTA": ("early_stop_min_delta", float),
        "RS_EARLY_STOP_PATIENCE": ("early_stop_patience", int),
        "RS_TRAIN_LOSS_STOP_ENABLED": ("train_loss_stop_enabled", parse_bool),
        "RS_TRAIN_LOSS_STOP_START_EPOCH": ("train_loss_stop_start_epoch", int),
        "RS_TRAIN_LOSS_STOP_MIN_DELTA": ("train_loss_stop_min_delta", float),
        "RS_TRAIN_LOSS_STOP_PATIENCE": ("train_loss_stop_patience", int),
        "RS_SUPPRESS_EXTERNAL_PROGRESS": ("suppress_external_progress", parse_bool),
    }

    for environment_name, (config_name, parser) in override_specs.items():
        if environment_name in os.environ:
            training_config[config_name] = parser(os.environ[environment_name])

    training_config.setdefault("suppress_external_progress", True)
    return training_config


def validate_training_control_config(training_config):
    if training_config["batch_size"] > training_config["max_batch_size"]:
        raise ValueError(
            f"BATCH_SIZE must be <= {training_config['max_batch_size']} for this experiment plan."
        )

    if training_config["validation_size"] < 0:
        raise ValueError("VALIDATION_SIZE must be greater than or equal to 0.")

    if training_config["early_stop_patience"] < 1:
        raise ValueError("EARLY_STOP_PATIENCE must be greater than or equal to 1.")

    if training_config["early_stop_min_delta"] < 0:
        raise ValueError("EARLY_STOP_MIN_DELTA must be greater than or equal to 0.")

    if training_config["early_stop_enabled"] and training_config["validation_size"] == 0:
        raise ValueError("VALIDATION_SIZE must be greater than 0 when early stopping is enabled.")

    if training_config.get("train_loss_stop_patience", 1) < 1:
        raise ValueError("TRAIN_LOSS_STOP_PATIENCE must be greater than or equal to 1.")

    if training_config.get("train_loss_stop_min_delta", 0) < 0:
        raise ValueError("TRAIN_LOSS_STOP_MIN_DELTA must be greater than or equal to 0.")


def create_train_validation_datasets(dataset, training_config):
    validation_size = training_config["validation_size"]

    if validation_size == 0:
        return dataset, None

    if validation_size >= len(dataset):
        raise ValueError("VALIDATION_SIZE must be smaller than the training dataset size.")

    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(training_config["validation_split_seed"])
    return random_split(dataset, (training_size, validation_size), generator=generator)


def should_suppress_external_progress(training_config):
    if not training_config["suppress_external_progress"]:
        return False

    return not sys.stdout.isatty() or not sys.stderr.isatty()


@contextlib.contextmanager
def suppress_external_progress_output(training_config):
    if not should_suppress_external_progress(training_config):
        yield
        return

    with open(os.devnull, "w", encoding=training_config["text_encoding"]) as null_file:
        with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
            yield


class EarlyStopping:
    def __init__(self, min_delta, patience):
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.best_epoch = None
        self.wait_count = 0

    def update(self, validation_loss, epoch):
        improved = self.best_loss is None or self.best_loss - validation_loss >= self.min_delta

        if improved:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.wait_count = 0
        else:
            self.wait_count += 1

        return {
            "improved": improved,
            "should_stop": self.wait_count >= self.patience,
            "best_val_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "wait_count": self.wait_count,
        }


def resolve_train_loss_stop_start_epoch(training_config):
    if training_config.get("train_loss_stop_start_epoch", 0) <= 0:
        training_config["train_loss_stop_start_epoch"] = training_config.get("warmup_epochs", 0) + 1


def update_train_loss_stopping(train_loss_stopping, average_loss, epoch, training_config):
    if not training_config.get("train_loss_stop_enabled", False):
        return create_inactive_train_loss_stop_state()

    if epoch < training_config["train_loss_stop_start_epoch"]:
        return create_inactive_train_loss_stop_state()

    stop_state = train_loss_stopping.update(average_loss, epoch)
    return {
        "active": True,
        "improved": stop_state["improved"],
        "should_stop": stop_state["should_stop"],
        "best_loss": stop_state["best_val_loss"],
        "best_epoch": stop_state["best_epoch"],
        "wait_count": stop_state["wait_count"],
    }


def create_inactive_train_loss_stop_state():
    return {
        "active": False,
        "improved": False,
        "should_stop": False,
        "best_loss": None,
        "best_epoch": None,
        "wait_count": 0,
    }
