import os
import torch

def save_model(args, model, filename='checkpoint.pth.tar'):

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)