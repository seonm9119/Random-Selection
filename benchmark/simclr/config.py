CIFAR10_DATASET_NAME = "cifar10"
CIFAR100_DATASET_NAME = "cifar100"
RESNET50_BACKBONE_NAME = "resnet50"

LEARNING_RATE_SCALING_LINEAR = "linear"
LEARNING_RATE_SCALING_SQRT = "sqrt"

PROJ_HEAD_MODE_NONE = "none"
PROJ_HEAD_MODE_LINEAR = "linear"
PROJ_HEAD_MODE_NONLINEAR = "nonlinear"

OPTIMIZER_NAME_MOMENTUM = "momentum"
OPTIMIZER_NAME_ADAM = "adam"
OPTIMIZER_NAME_LARS = "lars"

SUPPORTED_DATASETS = (CIFAR10_DATASET_NAME, CIFAR100_DATASET_NAME)
SUPPORTED_BACKBONES = (RESNET50_BACKBONE_NAME,)
SUPPORTED_PROJ_HEAD_MODES = (PROJ_HEAD_MODE_NONE, PROJ_HEAD_MODE_LINEAR, PROJ_HEAD_MODE_NONLINEAR)
SUPPORTED_OPTIMIZERS = (OPTIMIZER_NAME_MOMENTUM, OPTIMIZER_NAME_ADAM, OPTIMIZER_NAME_LARS)

PROJECT_DIR_PARENT_DEPTH = 2

REFERENCE_CONFIG = {
    "paper_reference": "A Simple Framework for Contrastive Learning of Visual Representations",
    "paper_url": "https://arxiv.org/abs/2002.05709",
    "official_github": "https://github.com/google-research/simclr",
    "paper_setting_source": "SimCLR Appendix B.9 CIFAR-10",
    "paper_epoch_options": (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
    "paper_batch_size_options": (256, 512, 1024, 2048, 4096),
    "paper_learning_rate_options": (0.5, 1.0, 1.5),
    "paper_temperature_options": (0.1, 0.5, 1.0),
    "paper_reported_best_batch_size": 1024,
    "paper_reported_linear_top1": 94.0,
}

RUN_CONFIG = {
    "dataset": CIFAR10_DATASET_NAME,
    "dataset_dir": "dataset",
    "output_dir": "benchmark/simclr/pretrained",
    "epochs": 1000,
    "batch_size": 1024,
    "planned_batch_sizes": (1024, 512, 256),
    "num_workers": 4,
    "seed": 0,
    "device": "auto",
    "amp": True,
    "max_batch_size": 1024,
}

DATASET_CONFIG = {
    "cifar10_dataset_name": CIFAR10_DATASET_NAME,
    "cifar100_dataset_name": CIFAR100_DATASET_NAME,
    "supported_datasets": SUPPORTED_DATASETS,
    "image_size": 32,
    "cifar_mean": (0.4914, 0.4822, 0.4465),
    "cifar_std": (0.2470, 0.2435, 0.2616),
    "normalize_images": False,
    "download_dataset": True,
    "train_split": True,
}

AUGMENTATION_CONFIG = {
    "color_strength": 0.5,
    "crop_scale": (0.08, 1.0),
    "crop_ratio": (0.75, 1.3333333333333333),
    "crop_interpolation": "bicubic",
    "horizontal_flip_probability": 0.5,
    "color_jitter_brightness": 0.8,
    "color_jitter_contrast": 0.8,
    "color_jitter_saturation": 0.8,
    "color_jitter_hue": 0.2,
    "color_jitter_probability": 0.8,
    "grayscale_probability": 0.2,
}

BACKBONE_CONFIG = {
    "backbone_name": RESNET50_BACKBONE_NAME,
    "resnet50_backbone_name": RESNET50_BACKBONE_NAME,
    "supported_backbones": SUPPORTED_BACKBONES,
    "resnet_weights": None,
    "resnet_first_conv_in_channels": 3,
    "resnet_first_conv_out_channels": 64,
    "resnet_first_conv_kernel_size": 3,
    "resnet_first_conv_stride": 1,
    "resnet_first_conv_padding": 1,
    "resnet_first_conv_bias": False,
    "backbone_feature_dims": {
        RESNET50_BACKBONE_NAME: 2048,
    },
}

PROJECTION_CONFIG = {
    "projection_hidden_dims": {
        RESNET50_BACKBONE_NAME: 2048,
    },
    "proj_head_mode_none": PROJ_HEAD_MODE_NONE,
    "proj_head_mode_linear": PROJ_HEAD_MODE_LINEAR,
    "proj_head_mode_nonlinear": PROJ_HEAD_MODE_NONLINEAR,
    "proj_head_mode": PROJ_HEAD_MODE_NONLINEAR,
    "supported_proj_head_modes": SUPPORTED_PROJ_HEAD_MODES,
    "proj_out_dim": 128,
    "num_proj_layers": 2,
    "projection_use_batch_norm": True,
    "projection_relu_inplace": False,
}

OPTIMIZER_CONFIG = {
    "optimizer_name_momentum": OPTIMIZER_NAME_MOMENTUM,
    "optimizer_name_adam": OPTIMIZER_NAME_ADAM,
    "optimizer_name_lars": OPTIMIZER_NAME_LARS,
    "optimizer_name": OPTIMIZER_NAME_LARS,
    "supported_optimizers": SUPPORTED_OPTIMIZERS,
    "learning_rate": 1.0,
    "learning_rate_scaling_linear": LEARNING_RATE_SCALING_LINEAR,
    "learning_rate_scaling_sqrt": LEARNING_RATE_SCALING_SQRT,
    "learning_rate_scaling": LEARNING_RATE_SCALING_LINEAR,
    "learning_rate_scale_reference_batch_size": 256,
    "warmup_epochs": 10,
    "weight_decay": 1e-6,
    "momentum": 0.9,
    "use_nesterov": True,
    "optimizer_set_to_none": True,
    "lars_eeta": 0.001,
    "lars_classic_momentum": True,
    "lars_exclude_from_weight_decay": ("batch_norm", "bn", "bias", "head_supervised"),
    "lars_exclude_from_layer_adaptation": ("batch_norm", "bn", "bias", "head_supervised"),
}

LOSS_CONFIG = {
    "temperature": 0.5,
    "view_count": 2,
    "batch_concat_dim": 0,
    "feature_normalize_dim": 1,
    "hidden_norm": True,
    "contrastive_large_num": 1e9,
    "contrastive_logit_concat_dim": 1,
    "tensor_sample_dim": 0,
}

DATALOADER_CONFIG = {
    "dataloader_shuffle": True,
    "dataloader_drop_last": True,
    "dataloader_pin_memory_with_cuda": True,
    "dataloader_persistent_workers": True,
    "min_worker_count_for_persistence": 0,
}

RUNTIME_CONFIG = {
    "project_dir_parent_depth": PROJECT_DIR_PARENT_DEPTH,
    "auto_device": "auto",
    "cuda_device": "cuda",
    "cpu_device": "cpu",
    "cudnn_benchmark": True,
    "training_start_epoch": 1,
    "step_start_index": 1,
    "epoch_loss_initial_value": 0.0,
    "official_train_steps_offset": 1,
}

ARTIFACT_CONFIG = {
    "config_file_name": "config.json",
    "checkpoint_file_template": "checkpoint_epoch_{epoch:04d}.pt",
    "best_checkpoint_file_name": "checkpoint_best.pt",
    "checkpoint_epoch_key": "epoch",
    "checkpoint_model_key": "model",
    "checkpoint_optimizer_key": "optimizer",
    "checkpoint_config_key": "training_config",
}

SERIALIZATION_CONFIG = {
    "json_indent": 2,
    "json_sort_keys": True,
    "text_encoding": "utf-8",
}

LOG_CONFIG = {
    "run_dir_log_template": "run_dir={run_dir}",
    "dataset_log_template": "dataset={dataset}",
    "device_log_template": "device={device}",
    "amp_log_template": "amp={amp}",
    "epoch_log_template": "epoch={epoch} train_loss={average_loss:.4f} lr={learning_rate:.6f}",
}

CONFIG_SECTIONS = (
    REFERENCE_CONFIG,
    RUN_CONFIG,
    DATASET_CONFIG,
    AUGMENTATION_CONFIG,
    BACKBONE_CONFIG,
    PROJECTION_CONFIG,
    OPTIMIZER_CONFIG,
    LOSS_CONFIG,
    DATALOADER_CONFIG,
    RUNTIME_CONFIG,
    ARTIFACT_CONFIG,
    SERIALIZATION_CONFIG,
    LOG_CONFIG,
)


def get_training_config():
    training_config = {}

    for config_section in CONFIG_SECTIONS:
        training_config.update(config_section)

    return training_config
