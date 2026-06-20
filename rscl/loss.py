import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomSelectionContrastiveLoss(nn.Module):
    def __init__(self, training_config):
        super().__init__()
        self.training_config = training_config

    def forward(self, projection_batch):
        if self.training_config["hidden_norm"]:
            projection_batch = F.normalize(projection_batch, dim=self.training_config["feature_normalize_dim"])

        logits = torch.matmul(projection_batch, projection_batch.T) / self.training_config["temperature"]
        batch_size = self.get_original_batch_size(projection_batch)
        positive_indices = self.create_positive_indices(batch_size, projection_batch.device)
        selected_negative_mask = self.create_selected_negative_mask(batch_size, projection_batch.device)

        anchor_indices = torch.arange(logits.shape[0], device=projection_batch.device)
        positive_logits = logits[anchor_indices, positive_indices]
        negative_logits = logits.masked_fill(~selected_negative_mask, -torch.inf)
        negative_logsumexp = torch.logsumexp(negative_logits, dim=1)
        scaled_negative_logsumexp = negative_logsumexp + math.log(self.training_config["negative_mass_scale"])
        denominator_logsumexp = torch.logaddexp(positive_logits, scaled_negative_logsumexp)
        return (denominator_logsumexp - positive_logits).mean()

    def get_original_batch_size(self, projection_batch):
        view_count = self.training_config["view_count"]
        total_sample_count = projection_batch.shape[self.training_config["tensor_sample_dim"]]

        if total_sample_count % view_count:
            raise ValueError("Projection batch size must be divisible by VIEW_COUNT.")

        batch_size = total_sample_count // view_count

        if batch_size < 2:
            raise ValueError("RSCL needs at least two original samples per batch.")

        return batch_size

    def create_positive_indices(self, batch_size, device):
        total_sample_count = batch_size * self.training_config["view_count"]
        sample_indices = torch.arange(total_sample_count, device=device)
        return (sample_indices + batch_size) % total_sample_count

    def create_selected_negative_mask(self, batch_size, device):
        total_sample_count = batch_size * self.training_config["view_count"]
        sample_indices = torch.arange(total_sample_count, device=device)
        positive_indices = self.create_positive_indices(batch_size, device)
        candidate_mask = sample_indices.unsqueeze(0) != sample_indices.unsqueeze(1)
        candidate_mask &= sample_indices.unsqueeze(0) != positive_indices.unsqueeze(1)
        candidate_count = int(candidate_mask[0].sum().item())
        selected_negative_count = self.get_selected_negative_count(candidate_count)

        if selected_negative_count == candidate_count:
            return candidate_mask

        random_scores = torch.rand(total_sample_count, total_sample_count, device=device)
        random_scores = random_scores.masked_fill(~candidate_mask, -torch.inf)
        selected_indices = random_scores.topk(selected_negative_count, dim=1).indices
        selected_negative_mask = torch.zeros_like(candidate_mask)
        selected_negative_mask.scatter_(1, selected_indices, True)
        return selected_negative_mask

    def get_selected_negative_count(self, candidate_count):
        configured_count = self.training_config["random_negative_count"]

        if configured_count is None:
            return candidate_count

        if configured_count <= 0:
            raise ValueError("RANDOM_NEGATIVE_COUNT must be positive or None.")

        return min(configured_count, candidate_count)
