import torch
import torch.nn as nn
import torch.nn.functional as F


class SimclrNtXentLoss(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.training_config = training_config

    def forward(self, projection_batch):
        if self.training_config["hidden_norm"]:
            projection_batch = F.normalize(projection_batch, dim=self.training_config["feature_normalize_dim"])

        split_size = projection_batch.shape[self.training_config["tensor_sample_dim"]] // self.training_config["view_count"]
        first_hidden_batch, second_hidden_batch = torch.split(
            projection_batch,
            split_size,
            dim=self.training_config["tensor_sample_dim"],
        )
        batch_size = first_hidden_batch.shape[self.training_config["tensor_sample_dim"]]
        labels = torch.arange(batch_size, device=projection_batch.device)
        masks = torch.eye(batch_size, device=projection_batch.device)

        logits_aa = torch.matmul(first_hidden_batch, first_hidden_batch.T) / self.training_config["temperature"]
        logits_aa = logits_aa - masks * self.training_config["contrastive_large_num"]
        logits_bb = torch.matmul(second_hidden_batch, second_hidden_batch.T) / self.training_config["temperature"]
        logits_bb = logits_bb - masks * self.training_config["contrastive_large_num"]
        logits_ab = torch.matmul(first_hidden_batch, second_hidden_batch.T) / self.training_config["temperature"]
        logits_ba = torch.matmul(second_hidden_batch, first_hidden_batch.T) / self.training_config["temperature"]

        loss_a = F.cross_entropy(
            torch.cat([logits_ab, logits_aa], dim=self.training_config["contrastive_logit_concat_dim"]),
            labels,
        )
        loss_b = F.cross_entropy(
            torch.cat([logits_ba, logits_bb], dim=self.training_config["contrastive_logit_concat_dim"]),
            labels,
        )
        return (loss_a + loss_b) * 0.5
