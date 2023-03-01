import torch
from torch import nn
import torch.nn.functional as F

# Cross Entropy Loss
# criterion = nn.CrossEntropyLoss()

# Cross Entropy Loss Class
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, 1), targets)


# Multi Focal Loss
class MultiFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss