import torch.nn as nn
import torch.nn.functional as F


class ByolRegressionLoss(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.training_config = training_config

    def forward(self, model_outputs):
        first_view_loss = self.calculate_regression_loss(
            model_outputs["online_prediction_first"],
            model_outputs["target_projection_second"],
        )
        second_view_loss = self.calculate_regression_loss(
            model_outputs["online_prediction_second"],
            model_outputs["target_projection_first"],
        )
        return (first_view_loss + second_view_loss).mean()

    def calculate_regression_loss(self, prediction_batch, target_projection_batch):
        prediction_batch = F.normalize(prediction_batch, dim=self.training_config["feature_normalize_dim"])
        target_projection_batch = F.normalize(
            target_projection_batch.detach(),
            dim=self.training_config["feature_normalize_dim"],
        )
        squared_distance = (prediction_batch - target_projection_batch).pow(2)
        return squared_distance.sum(dim=self.training_config["loss_reduction_dim"])
