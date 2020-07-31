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
from loadData import loadData
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, mean_squared_error

class AE(nn.Module):
    def __init__(self,hidden_dim):
        super(AE, self).__init__()
        self.fc4 = nn.Linear(784,hidden_dim)
        self.fc5 = nn.Linear(hidden_dim,784)
        

    def forward(self, x):
        x = torch.reshape(x,(-1,784))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.reshape(x,(-1,28,28))
        return x   


class Neural_Network():
    def __init__(self,train_loader,valid_loader,test_loader,device,size,hidden_dim):
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = train_loader,valid_loader,test_loader,device,size
        self.model =AE(hidden_dim)
        self.hidden_dim=hidden_dim
        self.model.to(self.device)
        self.epochs = 7
        self.lr = 5e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=0.01)
        self.writer = SummaryWriter(flush_secs =10)
        self.steps = 0
        self.logging_interval = 150

    def train(self):
        for epoch in range(0, self.epochs):
            self.train_epoch(epoch)
        self.test_epoch('test')

        return self.model, self.test_loader

    def train_epoch(self,epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.squeeze()
            target = data.squeeze()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            correct = 0
            loss = F.binary_cross_entropy(output, target,reduction='sum')
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('Loss/train', loss.item()/(784*len(data)),self.steps)
            self.steps +=1

            if batch_idx % self.logging_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f})%]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), self.size[0],
                    100.*batch_idx*len(data) /self.size[0], loss.item()/(784*len(data))))
                self.test_epoch('valid')

                self.model.train()

    def test_epoch(self,type_set):
        self.model.eval()
        curr_loader = self.valid_loader if type_set == 'valid' else self.test_loader
        size_value = 1 if type_set =='valid' else 2
        loss = 0
        with torch.no_grad():
            for data, target in curr_loader:
                data = data.squeeze()
                target = data.squeeze()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if type_set=='valid':
                    loss += F.binary_cross_entropy(output, target.squeeze(), reduction='sum').item()
                else:
                    loss += F.mse_loss(output, target.squeeze(), reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True) 
        loss /= 10000*784
        if type_set== 'valid':

            self.writer.add_scalar('Loss/valid',
                     loss,self.steps)

        print('{} set: Average loss: {:.4f}'.format(
            type_set,loss))
        with torch.no_grad():
            for data, target in curr_loader:
                data = data.squeeze()
                data = data.to(self.device)
                data_r = torch.rand((1,28,28)).cuda()
                output = self.model(data)
                output_r = self.model(data_r)
                data2 = data.cpu().detach().numpy()
                output2= output.cpu().detach().numpy()
                data2r = data_r.cpu().detach().numpy()
                output2r= output_r.cpu().detach().numpy()
                plt.imsave(f'.\\Progress\\{self.hidden_dim}orig.jpeg',data2[0,:,:])
                plt.imsave(f'.\\Progress\\{self.hidden_dim}back.jpeg',output2[0,:,:])
                plt.imsave(f'.\\Progress\\{self.hidden_dim}orig_r.jpeg',data2r[0,:,:])
                plt.imsave(f'.\\Progress\\{self.hidden_dim}back_r.jpeg',output2r[0,:,:])
                
                print(F.mse_loss(output.squeeze(), data.squeeze(), reduction='sum').item()/784)
                break



if __name__=='__main__':
    train_loader,valid_loader,test_loader,device,size = loadData()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dim',type=int)
    args = parser.parse_args()
    c = Neural_Network(train_loader,valid_loader,test_loader,device,size,args.dim)
    c.train()




