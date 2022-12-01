# Source: https://rwightman.github.io/pytorch-image-models/models/xception/

import timm 

model = timm.create_model('xception', pretrained=True)
model.eval()

