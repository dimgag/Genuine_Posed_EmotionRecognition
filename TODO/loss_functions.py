# Call Cross Entropy Loss

import torch
from torch import nn
import torch.nn.functional as F

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Cross Entropy Loss Class
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, 1), targets)


# Triple Loss -- PyTorch Example
# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
# anchor = torch.randn(100, 128, requires_grad=True)
# positive = torch.randn(100, 128, requires_grad=True)
# negative = torch.randn(100, 128, requires_grad=True)
# output = triplet_loss(anchor, positive, negative)
# output.backward()

# Triplet Loss // TripletMargin Loss
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        d_p = (anchor - positive).pow(self.p).sum(1)  # .pow(.5)
        d_n = (anchor - negative).pow(self.p).sum(1)  # .pow(.5)

        if self.swap:
            d_n_swapped = (positive - negative).pow(self.p).sum(1)  # .pow(.5)
            loss = torch.max(d_p - d_n_swapped + self.margin, torch.zeros_like(d_p))
        else:
            loss = torch.max(d_p - d_n + self.margin, torch.zeros_like(d_p))

        return loss.mean()



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


# Where to use Multi Focal Loss
# multi_focal_loss = MultiFocalLoss()
# output = multi_focal_loss(inputs, targets)
# output.backward()

