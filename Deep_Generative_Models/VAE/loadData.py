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
from torch.utils.data import TensorDataset


def loadData(batch_size=256):
    cuda_train = True
    device = torch.device("cuda" if cuda_train else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_train else {}
    train = np.arange(0,50000,1,dtype=int)
    val = np.arange(50000,60000,1,dtype=int)
    train_sampler = SubsetRandomSampler(train)
    valid_sampler = SubsetRandomSampler(val)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),batch_size= batch_size, shuffle=False,sampler=train_sampler,**kwargs)
    
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),batch_size= batch_size, shuffle=False,sampler=valid_sampler,**kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True,**kwargs)
    return train_loader,valid_loader,test_loader,device,(len(train_sampler),len(valid_sampler),len(test_loader.dataset))

