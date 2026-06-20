import re

import torch


class LarsOptimizer(torch.optim.Optimizer):
    def __init__(self, named_parameters, learning_rate, momentum, weight_decay, eeta=0.001):
        named_parameter_list = [
            (parameter_name, parameter)
            for parameter_name, parameter in named_parameters
            if parameter.requires_grad
        ]
        parameters = [parameter for _, parameter in named_parameter_list]
        parameter_names = [parameter_name for parameter_name, _ in named_parameter_list]
        defaults = {
            "lr": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "eeta": eeta,
            "parameter_names": parameter_names,
            "exclude_patterns": ("batch_norm", "bn", "bias"),
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for parameter_group in self.param_groups:
            self.update_parameter_group(parameter_group)

        return loss

    def update_parameter_group(self, parameter_group):
        parameter_names = parameter_group["parameter_names"]

        for parameter_index, parameter in enumerate(parameter_group["params"]):
            if parameter.grad is None:
                continue

            parameter_name = parameter_names[parameter_index]
            gradient = parameter.grad

            if self.should_use_weight_decay(parameter_name, parameter_group):
                gradient = gradient.add(parameter, alpha=parameter_group["weight_decay"])

            parameter_state = self.state[parameter]

            if "momentum_buffer" not in parameter_state:
                parameter_state["momentum_buffer"] = torch.zeros_like(parameter)

            trust_ratio = self.calculate_trust_ratio(parameter, gradient, parameter_name, parameter_group)
            scaled_learning_rate = parameter_group["lr"] * trust_ratio
            momentum_buffer = parameter_state["momentum_buffer"]
            momentum_buffer.mul_(parameter_group["momentum"]).add_(gradient, alpha=scaled_learning_rate)
            parameter.add_(momentum_buffer, alpha=-1.0)

    def calculate_trust_ratio(self, parameter, gradient, parameter_name, parameter_group):
        if self.matches_any_pattern(parameter_name, parameter_group["exclude_patterns"]):
            return 1.0

        parameter_norm = torch.norm(parameter)
        gradient_norm = torch.norm(gradient)

        if parameter_norm <= 0 or gradient_norm <= 0:
            return 1.0

        return (parameter_group["eeta"] * parameter_norm / gradient_norm).item()

    def should_use_weight_decay(self, parameter_name, parameter_group):
        if not parameter_group["weight_decay"]:
            return False

        return not self.matches_any_pattern(parameter_name, parameter_group["exclude_patterns"])

    def matches_any_pattern(self, parameter_name, patterns):
        return any(re.search(pattern, parameter_name) is not None for pattern in patterns)


def create_linear_optimizer(classifier, optimizer_name, learning_rate, momentum, weight_decay):
    if optimizer_name == "lars":
        return LarsOptimizer(classifier.named_parameters(), learning_rate, momentum, weight_decay)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            classifier.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported linear optimizer: {optimizer_name}")
