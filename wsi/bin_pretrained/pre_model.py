import numpy as np

np.random.seed(0)

import torch

from torchvision.models.resnet import (resnet34, resnet18, resnet50, resnet101, resnet152)

MODELS = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101,
          'resnet152': resnet152}


class resnet_crf(torch.nn.Module):
    def __init__(self, key, grid_size, pretrained, num_class=1):
        super(resnet_crf, self).__init__()
        model = MODELS[key](pretrained=pretrained)
        num_fc_ftr = model.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = torch.nn.Linear(num_fc_ftr, num_class)
        self.crf = CRF(grid_size)

    def forward(self, x):
        batch_size, grid_size, _, crop_size = x.shape[0:4]
        x = x.view(-1, 3, crop_size, crop_size)
        x = self.resnet(x)
        logits = self.fc(x)
        feats = feats.view((batch_size, grid_size, -1))
        logits = logits.view((batch_size, grid_size, -1))
        logits = self.crf(feats, logits)
        logits = torch.squeeze(logits)
        return logits


class resnet_base(torch.nn.Module):
    def __init__(self, key, pretrained, num_class=1):
        super(resnet_crf, self).__init__()
        self.model = MODELS[key](pretrained=pretrained)
        num_fc_ftr = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fc_ftr, num_class)

    def forward(self, x):
        batch_size, grid_size, _, crop_size = x.shape[0:4]
        x = x.view(-1, 3, crop_size, crop_size)
        logits = self.resnet(x)
        logits = logits.view((batch_size, grid_size, -1))
        logits = torch.squeeze(logits)
        return logits


