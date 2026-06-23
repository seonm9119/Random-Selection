import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from common.checkpoints import get_checkpoint_training_config, load_checkpoint, load_model_state


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "simclr": {
        "config_path": PROJECT_DIR / "benchmark" / "simclr" / "config.py",
        "model_path": PROJECT_DIR / "benchmark" / "simclr" / "model.py",
        "model_class_name": "SimclrModel",
        "encoder_path": "encoder",
    },
    "rscl": {
        "config_path": PROJECT_DIR / "rscl" / "config.py",
        "model_path": PROJECT_DIR / "rscl" / "model.py",
        "model_class_name": "RsclModel",
        "encoder_path": "encoder",
    },
}


class EncoderFeatureExtractor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image_batch):
        return self.encoder(image_batch)


def get_supported_model_names():
    return tuple(MODEL_SPECS.keys())


def get_model_spec(model_name):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")

    return MODEL_SPECS[model_name]


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_default_training_config(model_name):
    model_spec = get_model_spec(model_name)
    module = load_module(f"{model_name}_config", model_spec["config_path"])
    return module.get_training_config()


def prepare_training_config(model_name, checkpoint_training_config=None):
    if checkpoint_training_config:
        training_config = dict(checkpoint_training_config)
    else:
        training_config = load_default_training_config(model_name)

    training_config["encoder_feature_dim"] = training_config["backbone_feature_dims"][training_config["backbone_name"]]

    if "projection_hidden_dims" in training_config:
        training_config["projection_hidden_dim"] = training_config["projection_hidden_dims"][
            training_config["backbone_name"]
        ]

    return training_config


def create_model(model_name, training_config):
    model_spec = get_model_spec(model_name)
    module = load_module(f"{model_name}_model", model_spec["model_path"])
    model_class = getattr(module, model_spec["model_class_name"])
    return model_class(training_config)


def get_nested_attribute(root_object, attribute_path):
    current_object = root_object

    for attribute_name in attribute_path.split("."):
        current_object = getattr(current_object, attribute_name)

    return current_object


def get_encoder(model_name, model):
    model_spec = get_model_spec(model_name)
    return get_nested_attribute(model, model_spec["encoder_path"])


def create_encoder_feature_extractor(model_name, model):
    return EncoderFeatureExtractor(get_encoder(model_name, model))


def load_pretrained_model(model_name, checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    training_config = prepare_training_config(model_name, get_checkpoint_training_config(checkpoint))
    model = create_model(model_name, training_config)
    load_model_state(model, checkpoint)
    model.to(device)
    model.eval()
    return model, training_config, checkpoint


def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    return torch.device(device_name)
