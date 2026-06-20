import math

import torch


def create_positive_indices(batch_size, device):
    total_sample_count = batch_size * 2
    sample_indices = torch.arange(total_sample_count, device=device)
    return (sample_indices + batch_size) % total_sample_count


def calculate_select_probability(candidate_count, false_negative_count, selected_negative_count):
    if false_negative_count <= 0 or selected_negative_count <= 0:
        return 0.0

    if false_negative_count > candidate_count:
        raise ValueError("False negative count cannot be larger than candidate count.")

    if selected_negative_count >= candidate_count:
        return 1.0

    true_negative_count = candidate_count - false_negative_count

    if true_negative_count < selected_negative_count:
        return 1.0

    log_no_false_negative = (
        math.lgamma(true_negative_count + 1)
        - math.lgamma(true_negative_count - selected_negative_count + 1)
        - math.lgamma(candidate_count + 1)
        + math.lgamma(candidate_count - selected_negative_count + 1)
    )
    return 1.0 - math.exp(log_no_false_negative)


def calculate_false_negative_stats(labels, selected_negative_counts):
    device = labels.device
    batch_size = labels.shape[0]
    view_labels = torch.cat([labels, labels], dim=0)
    total_sample_count = view_labels.shape[0]
    sample_indices = torch.arange(total_sample_count, device=device)
    positive_indices = create_positive_indices(batch_size, device)
    candidate_mask = sample_indices.unsqueeze(0) != sample_indices.unsqueeze(1)
    candidate_mask &= sample_indices.unsqueeze(0) != positive_indices.unsqueeze(1)
    same_label_mask = view_labels.unsqueeze(0) == view_labels.unsqueeze(1)
    false_negative_mask = same_label_mask & candidate_mask

    candidate_count = int(candidate_mask[0].sum().item())
    false_negative_counts = false_negative_mask.sum(dim=1).float()
    simclr_false_negative_count = false_negative_counts.mean().item()
    simclr_false_negative_ratio = simclr_false_negative_count / candidate_count
    model_stats = {
        "candidate_count": candidate_count,
        "simclr_false_negative_count": simclr_false_negative_count,
        "simclr_false_negative_ratio": simclr_false_negative_ratio,
    }

    rscl_stats = {}
    for selected_negative_count in selected_negative_counts:
        effective_selected_count = min(selected_negative_count, candidate_count)
        expected_counts = false_negative_counts * effective_selected_count / candidate_count
        selection_probabilities = [
            calculate_select_probability(candidate_count, int(false_negative_count.item()), effective_selected_count)
            for false_negative_count in false_negative_counts
        ]
        average_selection_probability = sum(selection_probabilities) / len(selection_probabilities)
        rscl_stats[str(selected_negative_count)] = {
            "effective_selected_negative_count": effective_selected_count,
            "expected_false_negative_count": expected_counts.mean().item(),
            "expected_false_negative_ratio": expected_counts.mean().item() / effective_selected_count,
            "false_negative_selection_probability": average_selection_probability,
            "exposure_fraction_vs_simclr": effective_selected_count / candidate_count,
        }

    model_stats["rscl"] = rscl_stats
    return model_stats
