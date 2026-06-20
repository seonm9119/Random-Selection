import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count):
        self.total += value * count
        self.count += count

    @property
    def average(self):
        if self.count == 0:
            return 0.0

        return self.total / self.count


def calculate_topk_accuracy(logits, labels, topk=(1,)):
    max_k = max(topk)
    _, predictions = logits.topk(max_k, dim=1)
    predictions = predictions.t()
    correct_predictions = predictions.eq(labels.reshape(1, -1).expand_as(predictions))
    accuracies = []

    for current_k in topk:
        correct_count = correct_predictions[:current_k].reshape(-1).float().sum(0)
        accuracies.append(correct_count.mul_(100.0 / labels.shape[0]).item())

    return accuracies
