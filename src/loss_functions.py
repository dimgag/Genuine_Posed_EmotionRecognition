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


# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_p = self.calc_euclidean(anchor, positive)
        d_n = self.calc_euclidean(anchor, negative)
        triplet_loss = torch.relu(d_p - d_n + self.margin)

        return triplet_loss.mean()

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

# Focal Loss

# def FL(x,y):
#     focal_loss = torch.hub.load(
# 	'adeelh/pytorch-multi-class-focal-loss',
# 	model='focal_loss',
# 	alpha=[.75, .25],
# 	gamma=2,
# 	reduction='mean',
# 	device='cpu',
# 	dtype=torch.float32,
# 	force_reload=False
#     )
#     loss = focal_loss(x,y)
#     return loss

