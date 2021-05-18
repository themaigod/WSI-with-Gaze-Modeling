import numpy as np

import torch

from torchvision.models.resnet import (resnet34, resnet18, resnet50, resnet101, resnet152)

np.random.seed(0)

MODELS = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101,
          'resnet152': resnet152}


class ResnetBase(torch.nn.Module):
    def __init__(self, key, pretrained, num_class=1):
        super(ResnetBase, self).__init__()
        model = MODELS[key](pretrained=pretrained)
        num_fc_ftr = model.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.num_fc_ftr = num_fc_ftr
        self.fc = torch.nn.Linear(num_fc_ftr, num_class)

    def forward(self, x):
        batch_size, grid_size, _, crop_size = x.shape[0:4]
        x = x.view(-1, 3, crop_size, crop_size)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x, batch_size


class ResnetGlobal(torch.nn.Module):
    def __init__(self, key, pretrained, num_class=1):
        super(ResnetGlobal, self).__init__()
        model = MODELS[key](pretrained=pretrained)
        num_fc_ftr = model.fc.in_features
        self.num_fc_ftr = num_fc_ftr
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = torch.nn.Linear(num_fc_ftr, num_class)

    def forward(self, x):
        batch_size, grid_size, _, crop_size = x.shape[0:4]
        x = x.view(-1, 3, crop_size, crop_size)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x


class ResnetAll(torch.nn.Module):
    def __init__(self, key, grid_size, pretrained, use_crf=0, num_class=1):
        super(ResnetAll, self).__init__()
        self.resnet_base = ResnetCrf(key, pretrained, num_class)
        self.resnet_global = ResnetGlobal(key, pretrained, num_class)
        num_fc_ftr = self.resnet_base.num_fc_ftr + self.resnet_global.num_fc_ftr
        self.fc = torch.nn.Linear(num_fc_ftr, num_class)
        self.use_crf = use_crf
        self.crf = CRF(grid_size)
        self.grid_size = grid_size

    def forward(self, x, y):
        x = self.resnet_base(x)
        y = self.resnet_global(y)
        xy = torch.cat([x, y], 1)
        xy = self.fc(xy)
        xy = xy.view((batch_size, self.grid_size, -1))
        x = x.view((batch_size, self.grid_size, -1))
        if self.use_crf:
            xy = self.crf(x, xy)
        xy = torch.squeeze(xy)
        return xy
