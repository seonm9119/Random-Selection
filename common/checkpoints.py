import torch


def load_checkpoint(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def get_checkpoint_training_config(checkpoint):
    return checkpoint.get("training_config", {})


def get_checkpoint_model_state(checkpoint):
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain a model state.")

    return checkpoint["model"]


def load_model_state(model, checkpoint):
    model.load_state_dict(get_checkpoint_model_state(checkpoint))
    return model
