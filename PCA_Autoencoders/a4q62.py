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
    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels =8,kernel_size= 3, stride = 1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,kernel_size= 3, stride = 1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels =16,kernel_size= 3, stride = 1,padding=1)

        self.conv4 = nn.ConvTranspose2d(in_channels=16,out_channels =16,kernel_size=3, stride = 3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=8,kernel_size= 3, stride = 1,padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=8,out_channels =1,kernel_size= 2, stride = 2,padding=0)
        

    def forward(self, x):
        x = torch.reshape(x,(-1,1,28,28))
        batch_size=x.shape[0]
        x = F.relu(self.conv1(x))
        x,i1 = F.max_pool2d(x,(2, 2),return_indices=True)
        x = F.relu(self.conv2(x))
        x,i2 = F.max_pool2d(x,(2, 2),return_indices=True)
        x = F.relu(self.conv3(x))
        enc,i3 = F.max_pool2d(x,(2, 2),return_indices=True)
        x = F.relu(self.conv4(enc))
        x = F.max_unpool2d(x,i2,kernel_size=2,output_size=(batch_size,16,14,14))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return torch.reshape(x,(-1,28,28))


class Neural_Network():
    def __init__(self,train_loader,valid_loader,test_loader,device,size):
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = train_loader,valid_loader,test_loader,device,size
        self.model =AE()
        self.model.to(self.device)
        self.epochs = 12
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
                output = self.model(data)
                data2 = data.cpu().detach().numpy()
                output2= output.cpu().detach().numpy()
                plt.imsave(f'.\\Progress62\\orig.png',data2[0,:,:])
                plt.imsave(f'.\\Progress62\\back.png',output2[0,:,:])
                s = (self.model.conv5.weight.data)
                print(s.shape)
                for i in range(5):
                    plt.imsave(f'.\\Progress62\\w{i}.png',s.cpu().detach().numpy()[i,i,:,:])             
                print(F.binary_cross_entropy(output.squeeze(), data.squeeze(), reduction='sum').item()/(256*784))
                break



if __name__=='__main__':
    train_loader,valid_loader,test_loader,device,size = loadData()
    c = Neural_Network(train_loader,valid_loader,test_loader,device,size)
    c.train()




