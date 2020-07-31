from torchsummary import summary
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial

class CNN(nn.Module):
    def __init__(self,batchnorm=False):
        super(CNN, self).__init__()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.conv2_bn = nn.BatchNorm2d(32)
            self.fc1_bn = nn.BatchNorm1d(500)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels =32,kernel_size= 3, stride = 1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,kernel_size= 3, stride = 1,padding=1)
        self.fc1 = nn.Linear(16*98, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*98)
        x = (self.fc1(x))

        if self.batchnorm:
            x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def forward_logits(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*98)
        x = (self.fc1(x))

        if self.batchnorm:
            x = self.fc1_bn(x)
        x = F.relu(x)
        return self.fc2(x)
device = torch.device('cpu')

model = CNN().to(device)
summary(model.cuda(),input_size =(1,28,28))