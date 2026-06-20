import math
import re

import torch


class LarsOptimizer(torch.optim.Optimizer):
    def __init__(self, named_parameters, learning_rate, training_config):
        named_parameter_list = [
            (parameter_name, parameter)
            for parameter_name, parameter in named_parameters
            if parameter.requires_grad
        ]
        parameters = [parameter for _, parameter in named_parameter_list]
        parameter_names = [parameter_name for parameter_name, _ in named_parameter_list]

        defaults = {
            "lr": learning_rate,
            "momentum": training_config["momentum"],
            "weight_decay": training_config["weight_decay"],
            "use_nesterov": training_config["use_nesterov"],
            "classic_momentum": training_config["lars_classic_momentum"],
            "eeta": training_config["lars_eeta"],
            "parameter_names": parameter_names,
            "exclude_from_weight_decay": training_config["lars_exclude_from_weight_decay"],
            "exclude_from_layer_adaptation": training_config["lars_exclude_from_layer_adaptation"],
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.update_parameter_group(group)

        return loss

    def update_parameter_group(self, group):
        parameter_names = group["parameter_names"]

        for parameter_index, parameter in enumerate(group["params"]):
            if parameter.grad is None:
                continue

            parameter_name = parameter_names[parameter_index]
            gradient = parameter.grad

            if self.should_use_weight_decay(parameter_name, group["exclude_from_weight_decay"]):
                gradient = gradient.add(parameter, alpha=group["weight_decay"])

            update = self.calculate_update(parameter, gradient, parameter_name, group)
            parameter.add_(update, alpha=-1.0)

    def calculate_update(self, parameter, gradient, parameter_name, group):
        parameter_state = self.state[parameter]

        if "momentum_buffer" not in parameter_state:
            parameter_state["momentum_buffer"] = torch.zeros_like(parameter)

        momentum_buffer = parameter_state["momentum_buffer"]

        if group["classic_momentum"]:
            trust_ratio = self.calculate_trust_ratio(parameter, gradient, parameter_name, group)
            scaled_learning_rate = group["lr"] * trust_ratio
            momentum_buffer.mul_(group["momentum"]).add_(gradient, alpha=scaled_learning_rate)

            if group["use_nesterov"]:
                return momentum_buffer.mul(group["momentum"]).add(gradient, alpha=scaled_learning_rate)

            return momentum_buffer

        momentum_buffer.mul_(group["momentum"]).add_(gradient)
        update = momentum_buffer

        if group["use_nesterov"]:
            update = momentum_buffer.mul(group["momentum"]).add(gradient)

        trust_ratio = self.calculate_trust_ratio(parameter, update, parameter_name, group)
        return update.mul(group["lr"] * trust_ratio)

    def calculate_trust_ratio(self, parameter, update, parameter_name, group):
        if not self.should_do_layer_adaptation(parameter_name, group["exclude_from_layer_adaptation"]):
            return 1.0

        parameter_norm = torch.norm(parameter)
        update_norm = torch.norm(update)

        if parameter_norm <= 0 or update_norm <= 0:
            return 1.0

        return (group["eeta"] * parameter_norm / update_norm).item()

    def should_use_weight_decay(self, parameter_name, exclude_from_weight_decay):
        if not self.defaults["weight_decay"]:
            return False

        return not self.matches_any_pattern(parameter_name, exclude_from_weight_decay)

    def should_do_layer_adaptation(self, parameter_name, exclude_from_layer_adaptation):
        return not self.matches_any_pattern(parameter_name, exclude_from_layer_adaptation)

    def matches_any_pattern(self, parameter_name, patterns):
        return any(re.search(pattern, parameter_name) is not None for pattern in patterns)


class WarmupCosineLearningRate:
    def __init__(self, training_config):
        self.training_config = training_config

    def get_learning_rate(self, global_step):
        scaled_learning_rate = self.training_config["scaled_learning_rate"]
        warmup_steps = self.training_config["warmup_steps"]

        if warmup_steps and global_step < warmup_steps:
            return global_step / warmup_steps * scaled_learning_rate

        decay_steps = max(1, self.training_config["total_train_steps"] - warmup_steps)
        decay_progress = min(1.0, (global_step - warmup_steps) / decay_steps)
        return scaled_learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def create_optimizer(model, training_config):
    if training_config["optimizer_name"] == training_config["optimizer_name_lars"]:
        return LarsOptimizer(model.named_parameters(), training_config["scaled_learning_rate"], training_config)

    if training_config["optimizer_name"] == training_config["optimizer_name_momentum"]:
        return torch.optim.SGD(
            model.parameters(),
            lr=training_config["scaled_learning_rate"],
            momentum=training_config["momentum"],
            weight_decay=training_config["weight_decay"],
            nesterov=training_config["use_nesterov"],
        )

    if training_config["optimizer_name"] == training_config["optimizer_name_adam"]:
        return torch.optim.Adam(
            model.parameters(),
            lr=training_config["scaled_learning_rate"],
            weight_decay=training_config["weight_decay"],
        )

    raise ValueError(f"Unknown optimizer {training_config['optimizer_name']}.")


def create_learning_rate_schedule(training_config):
    return WarmupCosineLearningRate(training_config)


def set_optimizer_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
