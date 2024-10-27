import torch
from torch import nn

class OHEM_CELoss(nn.Module):
    def __init__(self, ratio=0.5, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', **kwargs)

    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        num = int(self.ratio * loss.size(0))
        loss, _ = loss.topk(num)
        return loss.mean()