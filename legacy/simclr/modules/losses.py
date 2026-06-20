import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature


    def forward(self, features):

        mask = torch.eq(features, features.T).float().cuda()

        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        logits_mask = ~torch.eye(features.shape[0], dtype=torch.bool).cuda()
        logits_mask = logits_mask.type(torch.FloatTensor).cuda()

        logits = logits * logits_mask
        exp_logits = torch.exp(logits) * logits_mask

        pos_mask = mask.repeat(2, 2) * logits_mask
        neg_mask = logits_mask - pos_mask

        exp_pos = exp_logits * pos_mask
        exp_pos_sum = torch.sum(exp_pos, 1)


        prob = exp_pos_sum # ((exp_neg_candi_sum * exp_neg_sum) + exp_pos_sum)
        loss = -torch.log(prob).mean()

        return loss





