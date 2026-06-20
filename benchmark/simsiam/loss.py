import torch.nn as nn


class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.training_config = training_config
        self.cosine_similarity = nn.CosineSimilarity(dim=training_config["feature_normalize_dim"])

    def forward(self, first_prediction, second_prediction, first_projection, second_projection):
        first_loss = self.cosine_similarity(first_prediction, second_projection.detach()).mean()
        second_loss = self.cosine_similarity(second_prediction, first_projection.detach()).mean()
        return -(first_loss + second_loss) * self.training_config["loss_symmetry_scale"]
