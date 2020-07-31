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
        self.fc4 = nn.Linear(784,32)
        self.fc5 = nn.Linear(32,784)
        

    def forward(self, x):
        x = torch.reshape(x,(-1,784))
#        x = F.relu(self.fc1(x))
        x = F.relu(self.fc4(x))
#        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.reshape(x,(-1,28,28))
        return x   


class Neural_Network():
    def __init__(self,train_loader,valid_loader,test_loader,device,size):
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = train_loader,valid_loader,test_loader,device,size
        self.model =AE()
        self.model.to(self.device)
        self.epochs = 12
        self.lr = 2e-3
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
                print('Train Epoch: {} [{}/{} ({:.0f})%]\tLoss: {:.6f}\n'.format(
                    epoch+1, batch_idx * len(data), self.size[0],
                    100.*batch_idx*len(data) /self.size[0], loss.item()/(784*len(data))),self.steps)
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

        loss /= 10000*784
        if type_set== 'valid':

            self.writer.add_scalar('Loss/valid',
                     loss,self.steps)

        print('{} set: Average loss: {:.4f}'.format(
            type_set,loss))



if __name__=='__main__':
    train_loader,valid_loader,test_loader,device,size = loadData()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pca',action='store_true')
    args = parser.parse_args()
    if args.pca:
        data_list = []
        test_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data_list.append(data.numpy().squeeze())
        data_list = np.concatenate(data_list)
        data_list_final = np.reshape(data_list,(-1,28*28))
        for batch_idx, (data, target) in enumerate(test_loader):
            test_list.append(data.numpy().squeeze())
        test_list = np.concatenate(test_list)
        test_list_final = np.reshape(test_list,(-1,28*28))

        print(data_list_final.shape)
        pca = PCA(n_components=30)
        pca.fit(data_list_final)
        Xt = pca.transform(test_list_final)
        print(Xt.shape)
        Xr = pca.inverse_transform(Xt)
        print('Using PCA',mean_squared_error(test_list_final.flatten(),Xr.flatten()))
        Xr1 = np.reshape(Xr,(-1,28,28))
        print(test_list[0,15:20,15:20],Xr1[0,15:20,15:20])
    else:
        c = Neural_Network(train_loader,valid_loader,test_loader,device,size)
        model,test_loader = c.train()
        torch.save(model.state_dict(),'stdae1layer.pth')





