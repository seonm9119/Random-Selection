import torch.nn as nn
import torchvision


class BatchNorm1dNoAffine(nn.BatchNorm1d):
    def __init__(self, feature_dim):
        super().__init__(feature_dim, affine=False)


def get_backbone_builder(training_config):
    backbone_builders = {
        training_config["resnet18_backbone_name"]: torchvision.models.resnet18,
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


class ProjectionHead(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        feature_dim = training_config["encoder_feature_dim"]
        projector_dim = training_config["projector_dim"]
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=training_config["projector_relu_inplace"]),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=training_config["projector_relu_inplace"]),
            nn.Linear(feature_dim, projector_dim),
            BatchNorm1dNoAffine(projector_dim),
        )
        self.layers[6].bias.requires_grad = False

    def forward(self, feature_batch):
        return self.layers(feature_batch)


class PredictorHead(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        projector_dim = training_config["projector_dim"]
        predictor_dim = training_config["predictor_dim"]
        self.layers = nn.Sequential(
            nn.Linear(projector_dim, predictor_dim, bias=False),
            nn.BatchNorm1d(predictor_dim),
            nn.ReLU(inplace=training_config["predictor_relu_inplace"]),
            nn.Linear(predictor_dim, projector_dim),
        )

    def forward(self, projection_batch):
        return self.layers(projection_batch)


class SimSiamModel(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.encoder = CifarResNetEncoder(training_config)
        self.projector = ProjectionHead(training_config)
        self.predictor = PredictorHead(training_config)

    def forward(self, first_view_batch, second_view_batch):
        first_projection = self.projector(self.encoder(first_view_batch))
        second_projection = self.projector(self.encoder(second_view_batch))
        first_prediction = self.predictor(first_projection)
        second_prediction = self.predictor(second_projection)
        return first_prediction, second_prediction, first_projection.detach(), second_projection.detach()
