import math

import torch


class CosineLearningRate:
    def __init__(self, training_config):
        self.training_config = training_config

    def get_learning_rate(self, epoch):
        current_epoch = epoch - self.training_config["training_start_epoch"]
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * current_epoch / self.training_config["epochs"]))
        return self.training_config["scaled_learning_rate"] * cosine_factor


def create_optimizer(model, training_config):
    if training_config["fix_predictor_learning_rate"]:
        optimizer_parameters = [
            {
                "params": list(model.encoder.parameters()) + list(model.projector.parameters()),
                "fix_learning_rate": False,
            },
            {
                "params": model.predictor.parameters(),
                "fix_learning_rate": True,
            },
        ]
    else:
        optimizer_parameters = model.parameters()

    return torch.optim.SGD(
        optimizer_parameters,
        training_config["scaled_learning_rate"],
        momentum=training_config["momentum"],
        weight_decay=training_config["weight_decay"],
    )


def create_learning_rate_schedule(training_config):
    return CosineLearningRate(training_config)


def set_optimizer_learning_rate(optimizer, learning_rate, training_config):
    for parameter_group in optimizer.param_groups:
        if parameter_group.get("fix_learning_rate", False):
            parameter_group["lr"] = training_config["scaled_learning_rate"]
        else:
            parameter_group["lr"] = learning_rate
