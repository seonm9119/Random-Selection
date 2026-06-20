CIFAR10_DATASET_NAME = "cifar10"
CIFAR100_DATASET_NAME = "cifar100"
DATASET = CIFAR100_DATASET_NAME
DATASET_DIR = "dataset"
OUTPUT_DIR = "benchmark/byol/pretrained"
SUPPORTED_DATASETS = (CIFAR10_DATASET_NAME, CIFAR100_DATASET_NAME)
PAPER_REFERENCE = "Bootstrap Your Own Latent"
PAPER_URL = "https://arxiv.org/abs/2006.07733"
OFFICIAL_GITHUB = "https://github.com/deepmind/deepmind-research/tree/master/byol"

EPOCHS = 400
BATCH_SIZE = 1024
PAPER_EPOCH_OPTIONS = (100, 500)
PAPER_BATCH_SIZE_OPTIONS = (256, 512)
NUM_WORKERS = 4
IMAGE_SIZE = 32
SEED = 0
DEVICE = "auto"
AMP = True
SAVE_EVERY = 100
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

RESNET50_BACKBONE_NAME = "resnet50"
BACKBONE_NAME = RESNET50_BACKBONE_NAME
SUPPORTED_BACKBONES = (RESNET50_BACKBONE_NAME,)
RESNET_WEIGHTS = None
RESNET_FIRST_CONV_IN_CHANNELS = 3
RESNET_FIRST_CONV_OUT_CHANNELS = 64
RESNET_FIRST_CONV_KERNEL_SIZE = 3
RESNET_FIRST_CONV_STRIDE = 1
RESNET_FIRST_CONV_PADDING = 1
RESNET_FIRST_CONV_BIAS = False
BACKBONE_FEATURE_DIMS = {
    RESNET50_BACKBONE_NAME: 2048,
}

PROJECTOR_HIDDEN_SIZE = 4096
PROJECTOR_OUTPUT_SIZE = 256
PREDICTOR_HIDDEN_SIZE = 4096
MLP_FIRST_LINEAR_BIAS = True
MLP_LAST_LINEAR_BIAS = False
MLP_RELU_INPLACE = True

BASE_LEARNING_RATE = 0.45
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-6
MOMENTUM = 0.9
LARS_EETA = 0.001
LARS_CLASSIC_MOMENTUM = True
LARS_EXCLUDE_FROM_WEIGHT_DECAY = ("batch_norm", "bn", "bias")
LARS_EXCLUDE_FROM_LAYER_ADAPTATION = ("batch_norm", "bn", "bias")
OPTIMIZER_SET_TO_NONE = True

BASE_TARGET_EMA = 0.99
EMA_MAX_VALUE = 1.0

VIEW_COUNT = 2
BATCH_CONCAT_DIM = 0
FEATURE_NORMALIZE_DIM = 1
LOSS_REDUCTION_DIM = -1

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

MIN_WORKER_COUNT_FOR_PERSISTENCE = 0
EPOCH_LOSS_INITIAL_VALUE = 0.0
FINAL_EPOCH_OFFSET = 1
SAVE_EVERY_REMAINDER = 0
OFFICIAL_TRAIN_STEPS_OFFSET = 0

CONFIG_FILE_NAME = "config.json"
CHECKPOINT_FILE_TEMPLATE = "checkpoint_epoch_{epoch:04d}.pt"
BEST_CHECKPOINT_FILE_NAME = "checkpoint_best.pt"
CHECKPOINT_EPOCH_KEY = "epoch"
CHECKPOINT_MODEL_KEY = "model"
CHECKPOINT_OPTIMIZER_KEY = "optimizer"
CHECKPOINT_CONFIG_KEY = "training_config"

TRAIN_LOSS_STEP_TAG = "train/loss_step"
TRAIN_LOSS_EPOCH_TAG = "train/loss_epoch"
TRAIN_LEARNING_RATE_TAG = "train/learning_rate"
TRAIN_TARGET_EMA_TAG = "train/target_ema"
RUN_DIR_LOG_TEMPLATE = "run_dir={run_dir}"
DATASET_LOG_TEMPLATE = "dataset={dataset}"
DEVICE_LOG_TEMPLATE = "device={device}"
AMP_LOG_TEMPLATE = "amp={amp}"
EPOCH_LOG_TEMPLATE = (
    "epoch={epoch} train_loss={average_loss:.4f} val_loss={validation_loss:.4f} "
    "lr={learning_rate:.6f} ema={target_ema:.6f} best_val_loss={best_val_loss:.4f} "
    "early_stop_wait={early_stop_wait}"
)


def get_training_config():
    return {
        "paper_reference": PAPER_REFERENCE,
        "paper_url": PAPER_URL,
        "official_github": OFFICIAL_GITHUB,
        "dataset": DATASET,
        "cifar10_dataset_name": CIFAR10_DATASET_NAME,
        "cifar100_dataset_name": CIFAR100_DATASET_NAME,
        "dataset_dir": DATASET_DIR,
        "output_dir": OUTPUT_DIR,
        "supported_datasets": SUPPORTED_DATASETS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "paper_epoch_options": PAPER_EPOCH_OPTIONS,
        "paper_batch_size_options": PAPER_BATCH_SIZE_OPTIONS,
        "num_workers": NUM_WORKERS,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "device": DEVICE,
        "amp": AMP,
        "save_every": SAVE_EVERY,
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
        "encoder_feature_dim": BACKBONE_FEATURE_DIMS[BACKBONE_NAME],
        "projector_hidden_size": PROJECTOR_HIDDEN_SIZE,
        "projector_output_size": PROJECTOR_OUTPUT_SIZE,
        "predictor_hidden_size": PREDICTOR_HIDDEN_SIZE,
        "mlp_first_linear_bias": MLP_FIRST_LINEAR_BIAS,
        "mlp_last_linear_bias": MLP_LAST_LINEAR_BIAS,
        "mlp_relu_inplace": MLP_RELU_INPLACE,
        "base_learning_rate": BASE_LEARNING_RATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
        "lars_eeta": LARS_EETA,
        "lars_classic_momentum": LARS_CLASSIC_MOMENTUM,
        "lars_exclude_from_weight_decay": LARS_EXCLUDE_FROM_WEIGHT_DECAY,
        "lars_exclude_from_layer_adaptation": LARS_EXCLUDE_FROM_LAYER_ADAPTATION,
        "optimizer_set_to_none": OPTIMIZER_SET_TO_NONE,
        "base_target_ema": BASE_TARGET_EMA,
        "ema_max_value": EMA_MAX_VALUE,
        "view_count": VIEW_COUNT,
        "batch_concat_dim": BATCH_CONCAT_DIM,
        "feature_normalize_dim": FEATURE_NORMALIZE_DIM,
        "loss_reduction_dim": LOSS_REDUCTION_DIM,
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
        "min_worker_count_for_persistence": MIN_WORKER_COUNT_FOR_PERSISTENCE,
        "epoch_loss_initial_value": EPOCH_LOSS_INITIAL_VALUE,
        "final_epoch_offset": FINAL_EPOCH_OFFSET,
        "save_every_remainder": SAVE_EVERY_REMAINDER,
        "official_train_steps_offset": OFFICIAL_TRAIN_STEPS_OFFSET,
        "config_file_name": CONFIG_FILE_NAME,
        "checkpoint_file_template": CHECKPOINT_FILE_TEMPLATE,
        "best_checkpoint_file_name": BEST_CHECKPOINT_FILE_NAME,
        "checkpoint_epoch_key": CHECKPOINT_EPOCH_KEY,
        "checkpoint_model_key": CHECKPOINT_MODEL_KEY,
        "checkpoint_optimizer_key": CHECKPOINT_OPTIMIZER_KEY,
        "checkpoint_config_key": CHECKPOINT_CONFIG_KEY,
        "train_loss_step_tag": TRAIN_LOSS_STEP_TAG,
        "train_loss_epoch_tag": TRAIN_LOSS_EPOCH_TAG,
        "train_learning_rate_tag": TRAIN_LEARNING_RATE_TAG,
        "train_target_ema_tag": TRAIN_TARGET_EMA_TAG,
        "run_dir_log_template": RUN_DIR_LOG_TEMPLATE,
        "dataset_log_template": DATASET_LOG_TEMPLATE,
        "device_log_template": DEVICE_LOG_TEMPLATE,
        "amp_log_template": AMP_LOG_TEMPLATE,
        "epoch_log_template": EPOCH_LOG_TEMPLATE,
    }
