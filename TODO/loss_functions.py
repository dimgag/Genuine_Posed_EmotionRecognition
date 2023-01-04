# Call Cross Entropy Loss

import torch
from torch import nn

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()


# Triple Loss
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = triplet_loss(anchor, positive, negative)
output.backward()

# Multi Focal Loss
#

