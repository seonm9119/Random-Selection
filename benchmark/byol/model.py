import copy

import torch
import torch.nn as nn
import torchvision


def get_backbone_builder(training_config):
    backbone_builders = {
        training_config["resnet50_backbone_name"]: torchvision.models.resnet50,
    }
    return backbone_builders[training_config["backbone_name"]]


class CifarResNetEncoder(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        backbone_builder = get_backbone_builder(training_config)
        self.backbone = backbone_builder(weights=training_config["resnet_weights"])
        self.backbone.conv1 = nn.Conv2d(
            training_config["resnet_first_conv_in_channels"],
            training_config["resnet_first_conv_out_channels"],
            kernel_size=training_config["resnet_first_conv_kernel_size"],
            stride=training_config["resnet_first_conv_stride"],
            padding=training_config["resnet_first_conv_padding"],
            bias=training_config["resnet_first_conv_bias"],
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.feature_dim = training_config["encoder_feature_dim"]

    def forward(self, image_batch):
        return self.backbone(image_batch)


class ByolMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, training_config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=training_config["mlp_first_linear_bias"]),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=training_config["mlp_relu_inplace"]),
            nn.Linear(hidden_dim, output_dim, bias=training_config["mlp_last_linear_bias"]),
        )

    def forward(self, feature_batch):
        return self.layers(feature_batch)


class ByolBranch(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.encoder = CifarResNetEncoder(training_config)
        self.projector = ByolMlp(
            training_config["encoder_feature_dim"],
            training_config["projector_hidden_size"],
            training_config["projector_output_size"],
            training_config,
        )

    def forward(self, image_batch):
        representation_batch = self.encoder(image_batch)
        projection_batch = self.projector(representation_batch)
        return representation_batch, projection_batch


class ByolModel(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.online_branch = ByolBranch(training_config)
        self.online_predictor = ByolMlp(
            training_config["projector_output_size"],
            training_config["predictor_hidden_size"],
            training_config["projector_output_size"],
            training_config,
        )
        self.target_branch = copy.deepcopy(self.online_branch)
        self.freeze_target_branch()

    def freeze_target_branch(self):
        for target_parameter in self.target_branch.parameters():
            target_parameter.requires_grad = False

    def forward(self, first_view_batch, second_view_batch):
        _, online_projection_first = self.online_branch(first_view_batch)
        _, online_projection_second = self.online_branch(second_view_batch)
        online_prediction_first = self.online_predictor(online_projection_first)
        online_prediction_second = self.online_predictor(online_projection_second)

        with torch.no_grad():
            _, target_projection_first = self.target_branch(first_view_batch)
            _, target_projection_second = self.target_branch(second_view_batch)

        return {
            "online_prediction_first": online_prediction_first,
            "online_prediction_second": online_prediction_second,
            "target_projection_first": target_projection_first.detach(),
            "target_projection_second": target_projection_second.detach(),
        }

    @torch.no_grad()
    def update_target_branch(self, target_ema):
        online_parameters = self.online_branch.parameters()
        target_parameters = self.target_branch.parameters()

        for online_parameter, target_parameter in zip(online_parameters, target_parameters):
            target_parameter.mul_(target_ema).add_(online_parameter, alpha=1.0 - target_ema)

        online_buffers = self.online_branch.buffers()
        target_buffers = self.target_branch.buffers()

        for online_buffer, target_buffer in zip(online_buffers, target_buffers):
            target_buffer.copy_(online_buffer)
