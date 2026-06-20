CIFAR10_DATASET_NAME = "cifar10"
CIFAR100_DATASET_NAME = "cifar100"
DATASET = CIFAR100_DATASET_NAME
DATASET_DIR = "dataset"
OUTPUT_DIR = "benchmark/simclr/pretrained"
SUPPORTED_DATASETS = (CIFAR10_DATASET_NAME, CIFAR100_DATASET_NAME)
PAPER_REFERENCE = "A Simple Framework for Contrastive Learning of Visual Representations"
PAPER_URL = "https://arxiv.org/abs/2002.05709"
OFFICIAL_GITHUB = "https://github.com/google-research/simclr"
PAPER_SETTING_SOURCE = "2022_KCC_Random_Selection_based_Loss_Function"

EPOCHS = 400
BATCH_SIZE = 1024
PAPER_EPOCH_OPTIONS = (100, 500)
PAPER_BATCH_SIZE_OPTIONS = (256, 512)
PAPER_DATASET_EXPERIMENTS = {
    CIFAR10_DATASET_NAME: {
        "batch_sizes": PAPER_BATCH_SIZE_OPTIONS,
        "epochs": PAPER_EPOCH_OPTIONS,
    },
    CIFAR100_DATASET_NAME: {
        "batch_sizes": PAPER_BATCH_SIZE_OPTIONS,
        "epochs": PAPER_EPOCH_OPTIONS,
    },
}
NUM_WORKERS = 4
IMAGE_SIZE = 32
TEMPERATURE = 0.5
LEARNING_RATE = 1.0
LEARNING_RATE_SCALING_LINEAR = "linear"
LEARNING_RATE_SCALING_SQRT = "sqrt"
LEARNING_RATE_SCALING = LEARNING_RATE_SCALING_LINEAR
LEARNING_RATE_SCALE_REFERENCE_BATCH_SIZE = 256
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-4
SEED = 0
DEVICE = "auto"
AMP = True
SAVE_EVERY = 100
SAVE_BEST_CHECKPOINT = True
MAX_BATCH_SIZE = 1024
VALIDATION_SIZE = 5000
VALIDATION_SPLIT_SEED = 0
EARLY_STOP_ENABLED = False
EARLY_STOP_MIN_DELTA = 3e-4
EARLY_STOP_PATIENCE = 5
TRAIN_LOSS_STOP_ENABLED = True
TRAIN_LOSS_STOP_START_EPOCH = 0
TRAIN_LOSS_STOP_MIN_DELTA = 0.01
TRAIN_LOSS_STOP_PATIENCE = 5

PROJECT_DIR_PARENT_DEPTH = 2
AUTO_DEVICE = "auto"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NORMALIZE_IMAGES = False
DOWNLOAD_DATASET = True
TRAIN_SPLIT = True

COLOR_STRENGTH = 0.5
CROP_SCALE = (0.08, 1.0)
CROP_RATIO = (0.75, 1.3333333333333333)
CROP_INTERPOLATION = "bicubic"
HORIZONTAL_FLIP_PROBABILITY = 0.5
COLOR_JITTER_BRIGHTNESS = 0.8
COLOR_JITTER_CONTRAST = 0.8
COLOR_JITTER_SATURATION = 0.8
COLOR_JITTER_HUE = 0.2
COLOR_JITTER_PROBABILITY = 0.8
GRAYSCALE_PROBABILITY = 0.2

RESNET18_BACKBONE_NAME = "resnet18"
RESNET50_BACKBONE_NAME = "resnet50"
BACKBONE_NAME = RESNET18_BACKBONE_NAME
SUPPORTED_BACKBONES = (RESNET18_BACKBONE_NAME, RESNET50_BACKBONE_NAME)
RESNET_WEIGHTS = None
RESNET_FIRST_CONV_IN_CHANNELS = 3
RESNET_FIRST_CONV_OUT_CHANNELS = 64
RESNET_FIRST_CONV_KERNEL_SIZE = 3
RESNET_FIRST_CONV_STRIDE = 1
RESNET_FIRST_CONV_PADDING = 1
RESNET_FIRST_CONV_BIAS = False
BACKBONE_FEATURE_DIMS = {
    RESNET18_BACKBONE_NAME: 512,
    RESNET50_BACKBONE_NAME: 2048,
}

PROJECTION_HIDDEN_DIMS = {
    RESNET18_BACKBONE_NAME: 512,
    RESNET50_BACKBONE_NAME: 2048,
}
PROJ_HEAD_MODE_NONE = "none"
PROJ_HEAD_MODE_LINEAR = "linear"
PROJ_HEAD_MODE_NONLINEAR = "nonlinear"
PROJ_HEAD_MODE = PROJ_HEAD_MODE_NONLINEAR
SUPPORTED_PROJ_HEAD_MODES = (PROJ_HEAD_MODE_NONE, PROJ_HEAD_MODE_LINEAR, PROJ_HEAD_MODE_NONLINEAR)
PROJ_OUT_DIM = 128
NUM_PROJ_LAYERS = 2
PROJECTION_USE_BATCH_NORM = True
PROJECTION_RELU_INPLACE = False

OPTIMIZER_NAME_MOMENTUM = "momentum"
OPTIMIZER_NAME_ADAM = "adam"
OPTIMIZER_NAME_LARS = "lars"
OPTIMIZER_NAME = OPTIMIZER_NAME_LARS
SUPPORTED_OPTIMIZERS = (OPTIMIZER_NAME_MOMENTUM, OPTIMIZER_NAME_ADAM, OPTIMIZER_NAME_LARS)
MOMENTUM = 0.9
USE_NESTEROV = True
OPTIMIZER_SET_TO_NONE = True
LARS_EETA = 0.001
LARS_CLASSIC_MOMENTUM = True
LARS_EXCLUDE_FROM_WEIGHT_DECAY = ("batch_norm", "bn", "bias", "head_supervised")
LARS_EXCLUDE_FROM_LAYER_ADAPTATION = ("batch_norm", "bn", "bias", "head_supervised")

VIEW_COUNT = 2
BATCH_CONCAT_DIM = 0
FEATURE_NORMALIZE_DIM = 1
HIDDEN_NORM = True
CONTRASTIVE_LARGE_NUM = 1e9
CONTRASTIVE_LOGIT_CONCAT_DIM = 1

DATALOADER_SHUFFLE = True
DATALOADER_DROP_LAST = True
DATALOADER_PIN_MEMORY_WITH_CUDA = True
DATALOADER_PERSISTENT_WORKERS = True

CUDNN_BENCHMARK = True
TRAINING_START_EPOCH = 1
STEP_START_INDEX = 1

JSON_INDENT = 2
JSON_SORT_KEYS = True
TEXT_ENCODING = "utf-8"

TENSOR_SAMPLE_DIM = 0
MIN_WORKER_COUNT_FOR_PERSISTENCE = 0
EPOCH_LOSS_INITIAL_VALUE = 0.0
FINAL_EPOCH_OFFSET = 1
SAVE_EVERY_REMAINDER = 0
PRIMARY_LEARNING_RATE_INDEX = 0
OFFICIAL_TRAIN_STEPS_OFFSET = 1

CONFIG_FILE_NAME = "config.json"
CHECKPOINT_FILE_TEMPLATE = "checkpoint_epoch_{epoch:04d}.pt"
BEST_CHECKPOINT_FILE_NAME = "checkpoint_best.pt"
CHECKPOINT_EPOCH_KEY = "epoch"
CHECKPOINT_MODEL_KEY = "model"
CHECKPOINT_OPTIMIZER_KEY = "optimizer"
CHECKPOINT_CONFIG_KEY = "training_config"

LINEAR_EVAL_LEARNING_RATE = 0.1
LINEAR_EVAL_OPTIMIZER_NAME = OPTIMIZER_NAME_LARS
LINEAR_EVAL_BATCH_SIZE = 512
LINEAR_EVAL_EPOCHS = 500

TRAIN_LOSS_STEP_TAG = "train/loss_step"
TRAIN_LOSS_EPOCH_TAG = "train/loss_epoch"
TRAIN_LEARNING_RATE_TAG = "train/learning_rate"
RUN_DIR_LOG_TEMPLATE = "run_dir={run_dir}"
DATASET_LOG_TEMPLATE = "dataset={dataset}"
DEVICE_LOG_TEMPLATE = "device={device}"
AMP_LOG_TEMPLATE = "amp={amp}"
EPOCH_LOG_TEMPLATE = (
    "epoch={epoch} train_loss={average_loss:.4f} val_loss={validation_loss:.4f} "
    "lr={learning_rate:.6f} best_val_loss={best_val_loss:.4f} early_stop_wait={early_stop_wait}"
)


def get_training_config():
    return {
        "dataset": DATASET,
        "paper_reference": PAPER_REFERENCE,
        "paper_url": PAPER_URL,
        "official_github": OFFICIAL_GITHUB,
        "paper_setting_source": PAPER_SETTING_SOURCE,
        "cifar10_dataset_name": CIFAR10_DATASET_NAME,
        "cifar100_dataset_name": CIFAR100_DATASET_NAME,
        "dataset_dir": DATASET_DIR,
        "output_dir": OUTPUT_DIR,
        "supported_datasets": SUPPORTED_DATASETS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "paper_epoch_options": PAPER_EPOCH_OPTIONS,
        "paper_batch_size_options": PAPER_BATCH_SIZE_OPTIONS,
        "paper_dataset_experiments": PAPER_DATASET_EXPERIMENTS,
        "num_workers": NUM_WORKERS,
        "image_size": IMAGE_SIZE,
        "temperature": TEMPERATURE,
        "learning_rate": LEARNING_RATE,
        "learning_rate_scaling_linear": LEARNING_RATE_SCALING_LINEAR,
        "learning_rate_scaling_sqrt": LEARNING_RATE_SCALING_SQRT,
        "learning_rate_scaling": LEARNING_RATE_SCALING,
        "learning_rate_scale_reference_batch_size": LEARNING_RATE_SCALE_REFERENCE_BATCH_SIZE,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "device": DEVICE,
        "amp": AMP,
        "save_every": SAVE_EVERY,
        "save_best_checkpoint": SAVE_BEST_CHECKPOINT,
        "max_batch_size": MAX_BATCH_SIZE,
        "validation_size": VALIDATION_SIZE,
        "validation_split_seed": VALIDATION_SPLIT_SEED,
        "early_stop_enabled": EARLY_STOP_ENABLED,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "train_loss_stop_enabled": TRAIN_LOSS_STOP_ENABLED,
        "train_loss_stop_start_epoch": TRAIN_LOSS_STOP_START_EPOCH,
        "train_loss_stop_min_delta": TRAIN_LOSS_STOP_MIN_DELTA,
        "train_loss_stop_patience": TRAIN_LOSS_STOP_PATIENCE,
        "project_dir_parent_depth": PROJECT_DIR_PARENT_DEPTH,
        "auto_device": AUTO_DEVICE,
        "cuda_device": CUDA_DEVICE,
        "cpu_device": CPU_DEVICE,
        "cifar_mean": CIFAR_MEAN,
        "cifar_std": CIFAR_STD,
        "normalize_images": NORMALIZE_IMAGES,
        "download_dataset": DOWNLOAD_DATASET,
        "train_split": TRAIN_SPLIT,
        "color_strength": COLOR_STRENGTH,
        "crop_scale": CROP_SCALE,
        "crop_ratio": CROP_RATIO,
        "crop_interpolation": CROP_INTERPOLATION,
        "horizontal_flip_probability": HORIZONTAL_FLIP_PROBABILITY,
        "color_jitter_brightness": COLOR_JITTER_BRIGHTNESS,
        "color_jitter_contrast": COLOR_JITTER_CONTRAST,
        "color_jitter_saturation": COLOR_JITTER_SATURATION,
        "color_jitter_hue": COLOR_JITTER_HUE,
        "color_jitter_probability": COLOR_JITTER_PROBABILITY,
        "grayscale_probability": GRAYSCALE_PROBABILITY,
        "backbone_name": BACKBONE_NAME,
        "resnet18_backbone_name": RESNET18_BACKBONE_NAME,
        "resnet50_backbone_name": RESNET50_BACKBONE_NAME,
        "supported_backbones": SUPPORTED_BACKBONES,
        "resnet_weights": RESNET_WEIGHTS,
        "resnet_first_conv_in_channels": RESNET_FIRST_CONV_IN_CHANNELS,
        "resnet_first_conv_out_channels": RESNET_FIRST_CONV_OUT_CHANNELS,
        "resnet_first_conv_kernel_size": RESNET_FIRST_CONV_KERNEL_SIZE,
        "resnet_first_conv_stride": RESNET_FIRST_CONV_STRIDE,
        "resnet_first_conv_padding": RESNET_FIRST_CONV_PADDING,
        "resnet_first_conv_bias": RESNET_FIRST_CONV_BIAS,
        "backbone_feature_dims": BACKBONE_FEATURE_DIMS,
        "projection_hidden_dims": PROJECTION_HIDDEN_DIMS,
        "proj_head_mode_none": PROJ_HEAD_MODE_NONE,
        "proj_head_mode_linear": PROJ_HEAD_MODE_LINEAR,
        "proj_head_mode_nonlinear": PROJ_HEAD_MODE_NONLINEAR,
        "proj_head_mode": PROJ_HEAD_MODE,
        "supported_proj_head_modes": SUPPORTED_PROJ_HEAD_MODES,
        "proj_out_dim": PROJ_OUT_DIM,
        "num_proj_layers": NUM_PROJ_LAYERS,
        "projection_use_batch_norm": PROJECTION_USE_BATCH_NORM,
        "projection_relu_inplace": PROJECTION_RELU_INPLACE,
        "optimizer_name_momentum": OPTIMIZER_NAME_MOMENTUM,
        "optimizer_name_adam": OPTIMIZER_NAME_ADAM,
        "optimizer_name_lars": OPTIMIZER_NAME_LARS,
        "optimizer_name": OPTIMIZER_NAME,
        "supported_optimizers": SUPPORTED_OPTIMIZERS,
        "momentum": MOMENTUM,
        "use_nesterov": USE_NESTEROV,
        "optimizer_set_to_none": OPTIMIZER_SET_TO_NONE,
        "lars_eeta": LARS_EETA,
        "lars_classic_momentum": LARS_CLASSIC_MOMENTUM,
        "lars_exclude_from_weight_decay": LARS_EXCLUDE_FROM_WEIGHT_DECAY,
        "lars_exclude_from_layer_adaptation": LARS_EXCLUDE_FROM_LAYER_ADAPTATION,
        "view_count": VIEW_COUNT,
        "batch_concat_dim": BATCH_CONCAT_DIM,
        "feature_normalize_dim": FEATURE_NORMALIZE_DIM,
        "hidden_norm": HIDDEN_NORM,
        "contrastive_large_num": CONTRASTIVE_LARGE_NUM,
        "contrastive_logit_concat_dim": CONTRASTIVE_LOGIT_CONCAT_DIM,
        "dataloader_shuffle": DATALOADER_SHUFFLE,
        "dataloader_drop_last": DATALOADER_DROP_LAST,
        "dataloader_pin_memory_with_cuda": DATALOADER_PIN_MEMORY_WITH_CUDA,
        "dataloader_persistent_workers": DATALOADER_PERSISTENT_WORKERS,
        "cudnn_benchmark": CUDNN_BENCHMARK,
        "training_start_epoch": TRAINING_START_EPOCH,
        "step_start_index": STEP_START_INDEX,
        "json_indent": JSON_INDENT,
        "json_sort_keys": JSON_SORT_KEYS,
        "text_encoding": TEXT_ENCODING,
        "tensor_sample_dim": TENSOR_SAMPLE_DIM,
        "min_worker_count_for_persistence": MIN_WORKER_COUNT_FOR_PERSISTENCE,
        "epoch_loss_initial_value": EPOCH_LOSS_INITIAL_VALUE,
        "final_epoch_offset": FINAL_EPOCH_OFFSET,
        "save_every_remainder": SAVE_EVERY_REMAINDER,
        "primary_learning_rate_index": PRIMARY_LEARNING_RATE_INDEX,
        "official_train_steps_offset": OFFICIAL_TRAIN_STEPS_OFFSET,
        "config_file_name": CONFIG_FILE_NAME,
        "checkpoint_file_template": CHECKPOINT_FILE_TEMPLATE,
        "best_checkpoint_file_name": BEST_CHECKPOINT_FILE_NAME,
        "checkpoint_epoch_key": CHECKPOINT_EPOCH_KEY,
        "checkpoint_model_key": CHECKPOINT_MODEL_KEY,
        "checkpoint_optimizer_key": CHECKPOINT_OPTIMIZER_KEY,
        "checkpoint_config_key": CHECKPOINT_CONFIG_KEY,
        "linear_eval_learning_rate": LINEAR_EVAL_LEARNING_RATE,
        "linear_eval_optimizer_name": LINEAR_EVAL_OPTIMIZER_NAME,
        "linear_eval_batch_size": LINEAR_EVAL_BATCH_SIZE,
        "linear_eval_epochs": LINEAR_EVAL_EPOCHS,
        "train_loss_step_tag": TRAIN_LOSS_STEP_TAG,
        "train_loss_epoch_tag": TRAIN_LOSS_EPOCH_TAG,
        "train_learning_rate_tag": TRAIN_LEARNING_RATE_TAG,
        "run_dir_log_template": RUN_DIR_LOG_TEMPLATE,
        "dataset_log_template": DATASET_LOG_TEMPLATE,
        "device_log_template": DEVICE_LOG_TEMPLATE,
        "amp_log_template": AMP_LOG_TEMPLATE,
        "epoch_log_template": EPOCH_LOG_TEMPLATE,
    }
