# -*- coding: utf-8 -*-
"""
2D Unet-like architecture code in Pytorch
"""
import math
import numpy as np
from .layers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


from torch.nn.parameter import Parameter
class PopulationNormalization(nn.Module):
    """
    Population Normalization (PN) for federated learning
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(PopulationNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Population statistics as trainable parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Trainable parameters for population statistics
        self.pop_mean = Parameter(torch.zeros(num_features))
        self.pop_gamma = Parameter(torch.ones(num_features))  # gamma^2 (variance)
        
        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.pop_mean)
        nn.init.ones_(self.pop_gamma)
    
    def forward(self, x):
        # x shape: (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=[0, 2, 3] if len(x.shape) == 4 else [0, 2, 3, 4])
            batch_var = x.var(dim=[0, 2, 3] if len(x.shape) == 4 else [0, 2, 3, 4], unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            
            # Use population statistics for normalization
            mean = self.pop_mean
            var = self.pop_gamma
        else:
            # Use population statistics during inference
            mean = self.pop_mean
            var = self.pop_gamma
        
        # Normalize
        if len(x.shape) == 4:  # 2D
            x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        else:  # 3D
            x_norm = (x - mean[None, :, None, None, None]) / torch.sqrt(var[None, :, None, None, None] + self.eps)
        
        # Scale and shift
        if self.affine:
            if len(x.shape) == 4:
                x_norm = x_norm * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            else:
                x_norm = x_norm * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        
        return x_norm
    
    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)





class MyUpsample2(nn.Module):
    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    elif norm == 'pn':
        m = PopulationNormalization(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

    def forward(self, x, weights=None, layer_idx=None):

        if weights == None:
            weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']

        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        #layer 1 conv, bn
        x = conv2d(x, weight_1, bias_1)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = conv2d(x, weight_2, bias_2)
        y = self.bn2(y)
        y = relu(y)

        #layer 3 conv, bn
        z = conv2d(y, weight_3, bias_3)
        z = self.bn3(z)
        z = relu(z)

        return z

class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = MyUpsample2()
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev, weights=None, layer_idx=None):

        if weights == None:
            if not self.first:
                weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            if not self.first:
                weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']
            
        #layer 1 conv, bn, relu
        if not self.first:
            x = conv2d(x, weight_1, bias_1, )
            x = self.bn1(x)
            x = relu(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = conv2d(y, weight_2, bias_2, kernel_size=1, stride=1, padding=0)
        y = self.bn2(y)
        y = relu(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = conv2d(y, weight_3, bias_3)
        y = self.bn3(y)
        y = relu(y)

        return y


class Unet2D(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=1):
        super(Unet2D, self).__init__()

        self.convd1 = ConvD(c,     n, norm, first=True)
        self.convd2 = ConvD(n,   2*n, norm)
        self.convd3 = ConvD(2*n, 4*n, norm)
        self.convd4 = ConvD(4*n, 8*n, norm)
        self.convd5 = ConvD(8*n,16*n, norm)

        self.convu4 = ConvU(16*n, norm, first=True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg1 = nn.Conv2d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, weights=None):

        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y1_pred = conv2d(y1, self.seg1.weight, self.seg1.bias, kernel_size=None, stride=1, padding=0)

        return y1_pred, y1

