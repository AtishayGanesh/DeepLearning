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
from torchvision.utils import save_image
from functools import partial
from loadData import loadData
import argparse


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.hidden_dim=int(sys.argv[1])
        self.fc1 = nn.Linear(784,self.hidden_dim)
        self.fc2m = nn.Linear(self.hidden_dim,20)
        self.fc2s = nn.Linear(self.hidden_dim,20)
        self.fc3 = nn.Linear(20,self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim,784)
        self.ct = 0

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2m(h1), self.fc2s(h1)

    def param(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.param(mu, logvar)
        return self.decode(z), mu, logvar


class Neural_network():
    def __init__(self,lr=5e-4,epochs=15,logging_interval=25):
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = loadData()
        self.model = VAE()
        self.model.to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=0.0001)
        self.writer = SummaryWriter(flush_secs =10)
        self.steps = 0
        self.logging_interval = logging_interval
        self.loss = nn.BCELoss(reduction='sum')
        self.sample = torch.randn(64, 20).to(self.device)

    

    def loss_function(self,reconstr_x, x, mu, logvar):
        BCE = self.loss(reconstr_x, x.view(-1, 784) )

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE , KLD


    def train(self):
        for epoch in range(0, self.epochs):
            self.train_epoch(epoch)
        self.test_epoch('test',-1)

        return self.model, self.test_loader

    def train_epoch(self,epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.model.train()
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss1 , loss2 = self.loss_function(recon_batch, data, mu, logvar)
            train_loss1,train_loss2 = loss1.item(),loss2.item()
            loss = loss1+loss2
            
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('Loss/BCE', loss1.item()/(len(target)*784),self.steps)
            self.writer.add_scalar('Loss/KLD', loss2.item()/(len(target)*20),self.steps)
            self.steps +=1
            if batch_idx%50==0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss1.item()/(len(target)*784),loss2.item()/(len(target)*20)}')
        self.test_epoch('valid',epoch)

            
    def test_epoch(self,type_set,epoch):
        self.model.eval()
        curr_loader = self.valid_loader if type_set == 'valid' else self.test_loader
        size_value = 1 if type_set =='valid' else 2
        loss = 0
        with torch.no_grad():
            for data, target in curr_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                l = (self.loss_function(recon_batch, data, mu, logvar))
                loss += l[0].item()+l[1].item()
        loss /= self.size[size_value]
        if type_set== 'valid':
            self.writer.add_scalar('Loss/valid',
                     loss,self.steps)
        opsample = self.model.decode(self.sample).cpu()
        save_image(opsample.view(64, 1, 28, 28),
                   f'results/sample_{self.model.hidden_dim}_{epoch}.png')


if __name__=='__main__':
    c = Neural_network()
    model,test_loader = c.train()
