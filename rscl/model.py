import torch.nn as nn
import torchvision


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        del self.bias
        self.register_parameter("bias", None)


def get_backbone_builder(training_config):
    backbone_builders = {
        training_config["resnet50_backbone_name"]: torchvision.models.resnet50,
    }
    return backbone_builders[training_config["backbone_name"]]


def create_linear_layer(input_dim, output_dim, use_bias, use_batch_norm):
    layers = [
        nn.Linear(input_dim, output_dim, bias=use_bias and not use_batch_norm),
    ]

    if use_batch_norm:
        if use_bias:
            layers.append(nn.BatchNorm1d(output_dim))
        else:
            layers.append(BatchNorm1dNoBias(output_dim))

    return layers


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
        self.layers = self.create_projection_layers(training_config)

    def create_projection_layers(self, training_config):
        if training_config["proj_head_mode"] == training_config["proj_head_mode_none"]:
            return nn.Identity()

        if training_config["proj_head_mode"] == training_config["proj_head_mode_linear"]:
            return nn.Sequential(
                *create_linear_layer(
                    training_config["encoder_feature_dim"],
                    training_config["proj_out_dim"],
                    use_bias=False,
                    use_batch_norm=training_config["projection_use_batch_norm"],
                )
            )

        if training_config["proj_head_mode"] == training_config["proj_head_mode_nonlinear"]:
            projection_layers = []
            input_dim = training_config["encoder_feature_dim"]

            for layer_index in range(training_config["num_proj_layers"]):
                is_final_layer = layer_index == training_config["num_proj_layers"] - 1

                if is_final_layer:
                    output_dim = training_config["proj_out_dim"]
                    use_bias = False
                    apply_relu = False
                else:
                    output_dim = training_config["projection_hidden_dim"]
                    use_bias = True
                    apply_relu = True

                projection_layers.extend(
                    create_linear_layer(
                        input_dim,
                        output_dim,
                        use_bias=use_bias,
                        use_batch_norm=training_config["projection_use_batch_norm"],
                    )
                )

                if apply_relu:
                    projection_layers.append(nn.ReLU(inplace=training_config["projection_relu_inplace"]))

                input_dim = output_dim

            return nn.Sequential(*projection_layers)

        raise ValueError(f"Unknown projection head mode {training_config['proj_head_mode']}.")

    def forward(self, feature_batch):
        return self.layers(feature_batch)


class RsclModel(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.encoder = CifarResNetEncoder(training_config)
        self.projection_head = ProjectionHead(training_config)

    def forward(self, image_batch):
        feature_batch = self.encoder(image_batch)
        projection_batch = self.projection_head(feature_batch)
        return feature_batch, projection_batch
