import json
import random
import sys
from pathlib import Path

import config
import numpy as np
import torch
import torchvision
from loss import ByolRegressionLoss
from model import ByolModel
from optimizer import create_learning_rate_schedule, create_optimizer, create_target_ema_schedule, set_optimizer_learning_rate
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parents[config.PROJECT_DIR_PARENT_DEPTH]
sys.path.insert(0, str(PROJECT_DIR))

from common.training_control import (  # noqa: E402
    EarlyStopping,
    apply_environment_overrides,
    create_train_validation_datasets,
    resolve_train_loss_stop_start_epoch,
    suppress_external_progress_output,
    update_train_loss_stopping,
    validate_training_control_config,
)
from common.training_logs import reset_training_log, write_training_log  # noqa: E402


def get_crop_interpolation(training_config):
    interpolation_modes = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
    }
    return interpolation_modes[training_config["crop_interpolation"]]


class ByolPairTransform:
    def __init__(self, training_config):
        color_jitter = transforms.ColorJitter(
            brightness=training_config["color_jitter_brightness"] * training_config["color_strength"],
            contrast=training_config["color_jitter_contrast"] * training_config["color_strength"],
            saturation=training_config["color_jitter_saturation"] * training_config["color_strength"],
            hue=training_config["color_jitter_hue"] * training_config["color_strength"],
        )
        transform_layers = [
            transforms.RandomResizedCrop(
                size=training_config["image_size"],
                scale=training_config["crop_scale"],
                ratio=training_config["crop_ratio"],
                interpolation=get_crop_interpolation(training_config),
            ),
            transforms.RandomHorizontalFlip(p=training_config["horizontal_flip_probability"]),
            transforms.RandomApply([color_jitter], p=training_config["color_jitter_probability"]),
            transforms.RandomGrayscale(p=training_config["grayscale_probability"]),
            transforms.ToTensor(),
        ]

        if training_config["normalize_images"]:
            transform_layers.append(transforms.Normalize(training_config["cifar_mean"], training_config["cifar_std"]))

        self.training_transform = transforms.Compose(transform_layers)

    def __call__(self, image):
        first_view = self.training_transform(image)
        second_view = self.training_transform(image)
        return first_view, second_view


def set_random_seed(training_config):
    random.seed(training_config["seed"])
    np.random.seed(training_config["seed"])
    torch.manual_seed(training_config["seed"])
    torch.cuda.manual_seed_all(training_config["seed"])
    torch.backends.cudnn.benchmark = training_config["cudnn_benchmark"]


def resolve_project_path(path_text):
    configured_path = Path(path_text)

    if configured_path.is_absolute():
        return str(configured_path)

    return str(PROJECT_DIR / configured_path)


def create_run_dir(training_config):
    run_dir = Path(training_config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_training_config(run_dir, training_config):
    config_path = run_dir / training_config["config_file_name"]
    config_text = json.dumps(
        training_config,
        indent=training_config["json_indent"],
        sort_keys=training_config["json_sort_keys"],
    )
    config_path.write_text(config_text, encoding=training_config["text_encoding"])


def create_cifar_dataset(training_config, pair_transform):
    dataset_classes = {
        training_config["cifar10_dataset_name"]: torchvision.datasets.CIFAR10,
        training_config["cifar100_dataset_name"]: torchvision.datasets.CIFAR100,
    }
    dataset_class = dataset_classes[training_config["dataset"]]

    with suppress_external_progress_output(training_config):
        return dataset_class(
            root=training_config["dataset_dir"],
            train=training_config["train_split"],
            transform=pair_transform,
            download=training_config["download_dataset"],
        )


def create_dataloader(training_dataset, training_config, device, shuffle=None, drop_last=None):
    if shuffle is None:
        shuffle = training_config["dataloader_shuffle"]

    if drop_last is None:
        drop_last = training_config["dataloader_drop_last"]

    return DataLoader(
        training_dataset,
        batch_size=training_config["batch_size"],
        shuffle=shuffle,
        num_workers=training_config["num_workers"],
        pin_memory=training_config["dataloader_pin_memory_with_cuda"] and device.type == training_config["cuda_device"],
        drop_last=drop_last,
        persistent_workers=(
            training_config["dataloader_persistent_workers"]
            and training_config["num_workers"] > training_config["min_worker_count_for_persistence"]
        ),
    )


def save_checkpoint(run_dir, model, optimizer, epoch, training_config, checkpoint_file_name=None):
    if checkpoint_file_name is None:
        checkpoint_file_name = training_config["checkpoint_file_template"].format(epoch=epoch)

    checkpoint_path = run_dir / checkpoint_file_name
    torch.save(
        {
            training_config["checkpoint_epoch_key"]: epoch,
            training_config["checkpoint_model_key"]: model.state_dict(),
            training_config["checkpoint_optimizer_key"]: optimizer.state_dict(),
            training_config["checkpoint_config_key"]: training_config,
        },
        checkpoint_path,
    )


def resolve_training_config():
    training_config = apply_environment_overrides(config.get_training_config())

    if training_config["device"] == training_config["auto_device"]:
        if torch.cuda.is_available():
            training_config["device"] = training_config["cuda_device"]
        else:
            training_config["device"] = training_config["cpu_device"]

    if training_config["dataset"] not in training_config["supported_datasets"]:
        raise ValueError(f"DATASET must be one of {training_config['supported_datasets']}.")

    if training_config["backbone_name"] not in training_config["supported_backbones"]:
        raise ValueError(f"BACKBONE_NAME must be one of {training_config['supported_backbones']}.")

    if training_config["crop_interpolation"] not in ("bicubic", "bilinear"):
        raise ValueError("CROP_INTERPOLATION must be 'bicubic' or 'bilinear'.")

    validate_training_control_config(training_config)
    training_config["encoder_feature_dim"] = training_config["backbone_feature_dims"][training_config["backbone_name"]]
    training_config["dataset_dir"] = resolve_project_path(training_config["dataset_dir"])
    training_config["output_dir"] = resolve_project_path(training_config["output_dir"])
    training_config["amp"] = training_config["amp"] and training_config["device"] == training_config["cuda_device"]
    return training_config


def update_training_step_config(training_config, dataloader):
    training_config["steps_per_epoch"] = len(dataloader)
    training_config["warmup_steps"] = int(round(training_config["warmup_epochs"] * len(dataloader)))
    training_config["total_train_steps"] = (
        training_config["epochs"] * len(dataloader)
        + training_config["official_train_steps_offset"]
    )
    training_config["scaled_learning_rate"] = training_config["base_learning_rate"] * training_config["batch_size"] / 256
    resolve_train_loss_stop_start_epoch(training_config)


def train_one_epoch(model, criterion, dataloader, optimizer, learning_rate_schedule, target_ema_schedule, scaler, device, training_config, epoch):
    model.train()
    epoch_loss_sum = training_config["epoch_loss_initial_value"]
    last_learning_rate = training_config["scaled_learning_rate"]
    last_target_ema = training_config["base_target_ema"]

    for step_index, ((first_view_batch, second_view_batch), _) in enumerate(
        dataloader,
        start=training_config["step_start_index"],
    ):
        first_view_batch = first_view_batch.to(device, non_blocking=True)
        second_view_batch = second_view_batch.to(device, non_blocking=True)
        global_step = (epoch - training_config["training_start_epoch"]) * len(dataloader)
        global_step += step_index - training_config["step_start_index"]
        last_learning_rate = learning_rate_schedule.get_learning_rate(global_step)
        last_target_ema = target_ema_schedule.get_target_ema(global_step)
        set_optimizer_learning_rate(optimizer, last_learning_rate)

        optimizer.zero_grad(set_to_none=training_config["optimizer_set_to_none"])

        with torch.cuda.amp.autocast(enabled=training_config["amp"]):
            model_outputs = model(first_view_batch, second_view_batch)
            training_loss = criterion(model_outputs)

        scaler.scale(training_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_target_branch(last_target_ema)

        epoch_loss_sum += training_loss.item()

    return epoch_loss_sum / len(dataloader), last_learning_rate, last_target_ema


def evaluate_one_epoch(model, criterion, dataloader, device, training_config):
    model.eval()
    epoch_loss_sum = training_config["epoch_loss_initial_value"]

    with torch.no_grad():
        for (first_view_batch, second_view_batch), _ in dataloader:
            first_view_batch = first_view_batch.to(device, non_blocking=True)
            second_view_batch = second_view_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=training_config["amp"]):
                model_outputs = model(first_view_batch, second_view_batch)
                validation_loss = criterion(model_outputs)

            epoch_loss_sum += validation_loss.item()

    return epoch_loss_sum / len(dataloader)


def main():
    training_config = resolve_training_config()
    set_random_seed(training_config)

    device = torch.device(training_config["device"])
    run_dir = create_run_dir(training_config)
    pair_transform = ByolPairTransform(training_config)
    full_training_dataset = create_cifar_dataset(training_config, pair_transform)
    training_dataset, validation_dataset = create_train_validation_datasets(full_training_dataset, training_config)
    dataloader = create_dataloader(training_dataset, training_config, device)
    validation_dataloader = create_dataloader(
        validation_dataset,
        training_config,
        device,
        shuffle=False,
        drop_last=training_config["dataloader_drop_last"],
    )
    update_training_step_config(training_config, dataloader)
    write_training_config(run_dir, training_config)
    reset_training_log(run_dir, training_config)

    model = ByolModel(training_config).to(device)
    criterion = ByolRegressionLoss(training_config).to(device)
    optimizer = create_optimizer(model, training_config)
    learning_rate_schedule = create_learning_rate_schedule(training_config)
    target_ema_schedule = create_target_ema_schedule(training_config)
    scaler = torch.cuda.amp.GradScaler(enabled=training_config["amp"])
    early_stopping = EarlyStopping(
        training_config["early_stop_min_delta"],
        training_config["early_stop_patience"],
    )
    train_loss_stopping = EarlyStopping(
        training_config["train_loss_stop_min_delta"],
        training_config["train_loss_stop_patience"],
    )

    print(training_config["run_dir_log_template"].format(run_dir=run_dir))
    print(training_config["dataset_log_template"].format(dataset=training_config["dataset"]))
    print(training_config["device_log_template"].format(device=device))
    print(training_config["amp_log_template"].format(amp=training_config["amp"]))

    last_epoch = training_config["epochs"] + training_config["training_start_epoch"]
    final_epoch = last_epoch - training_config["final_epoch_offset"]

    for epoch in range(training_config["training_start_epoch"], last_epoch):
        average_loss, learning_rate, target_ema = train_one_epoch(
            model,
            criterion,
            dataloader,
            optimizer,
            learning_rate_schedule,
            target_ema_schedule,
            scaler,
            device,
            training_config,
            epoch,
        )
        validation_loss = evaluate_one_epoch(model, criterion, validation_dataloader, device, training_config)
        early_stop_state = early_stopping.update(validation_loss, epoch)
        train_loss_stop_state = update_train_loss_stopping(train_loss_stopping, average_loss, epoch, training_config)
        write_training_log(
            run_dir,
            training_config,
            {
                "epoch": epoch,
                "loss": average_loss,
                "train_loss": average_loss,
                "val_loss": validation_loss,
                "learning_rate": learning_rate,
                "target_ema": target_ema,
                "best_val_loss": early_stop_state["best_val_loss"],
                "best_epoch": early_stop_state["best_epoch"],
                "early_stop_wait": early_stop_state["wait_count"],
                "early_stop_improved": early_stop_state["improved"],
                "train_loss_stop_active": train_loss_stop_state["active"],
                "train_loss_stop_best_loss": train_loss_stop_state["best_loss"],
                "train_loss_stop_best_epoch": train_loss_stop_state["best_epoch"],
                "train_loss_stop_wait": train_loss_stop_state["wait_count"],
                "train_loss_stop_improved": train_loss_stop_state["improved"],
                "train_loss_stop_patience": training_config["train_loss_stop_patience"],
            },
        )

        print(
            training_config["epoch_log_template"].format(
                epoch=epoch,
                average_loss=average_loss,
                validation_loss=validation_loss,
                learning_rate=learning_rate,
                target_ema=target_ema,
                best_val_loss=early_stop_state["best_val_loss"],
                early_stop_wait=early_stop_state["wait_count"],
            )
        )

        if early_stop_state["improved"]:
            save_checkpoint(
                run_dir,
                model,
                optimizer,
                epoch,
                training_config,
                training_config["best_checkpoint_file_name"],
            )

        if epoch % training_config["save_every"] == training_config["save_every_remainder"] or epoch == final_epoch:
            save_checkpoint(run_dir, model, optimizer, epoch, training_config)

        if train_loss_stop_state["should_stop"]:
            print(
                "train_loss_stop "
                f"epoch={epoch} best_epoch={train_loss_stop_state['best_epoch']} "
                f"best_train_loss={train_loss_stop_state['best_loss']:.4f}"
            )
            break

        if training_config["early_stop_enabled"] and early_stop_state["should_stop"]:
            save_checkpoint(run_dir, model, optimizer, epoch, training_config)
            print(
                "early_stop "
                f"epoch={epoch} best_epoch={early_stop_state['best_epoch']} "
                f"best_val_loss={early_stop_state['best_val_loss']:.4f}"
            )
            break


if __name__ == "__main__":
    main()
