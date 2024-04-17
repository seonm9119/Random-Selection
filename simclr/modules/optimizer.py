import torch
import numpy as np
import warnings
import math

from simclr.modules import LARS
from simclr.modules import CosineAnnealingWarmupRestarts


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]


def load_optimizers(args, model, num_examples, cur_iter=-1):


    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if args.optimizer == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]



    LR = args.lr * args.batch_size/256.

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=LR)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=LR, weight_decay=args.weight_decay,
                          exclude_from_weight_decay=["batch_normalization", "bias"])
    else:
        raise NotImplementedError

    if args.lr_schedule == 'warmup-anneal':
        scheduler = LinearWarmupAndCosineAnneal(optimizer, args.warmup, args.epochs, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = None
    else:
        raise NotImplementedError

    if args.optimizer == 'lars':
        optimizer = larc_optimizer


    return optimizer, scheduler
