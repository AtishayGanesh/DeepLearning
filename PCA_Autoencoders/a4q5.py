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
class stdAE(nn.Module):
    def __init__(self):
        super(stdAE, self).__init__()
        self.fc1 = nn.Linear(784,64)
        self.fc4 = nn.Linear(64,8)

        self.fc5 = nn.Linear(8,64)
        self.fc7 = nn.Linear(64,784)
    def forward(self, x):

        return self.decode(self.encode(x))
    def encode(self,x):
        x = torch.reshape(x,(-1,784))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc4(x))
        return x

    def decode(self,x):
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc7(x))
        x = torch.reshape(x,(-1,28,28))
        return x



class Neural_Network():
    def __init__(self,train_loader,valid_loader,test_loader,device,size,hidden_dim):
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = train_loader,valid_loader,test_loader,device,size
        self.model =AE(hidden_dim)
        self.hidden_dim=hidden_dim
        self.model.to(self.device)
        self.epochs = 10
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=0.01)
        self.writer = SummaryWriter(flush_secs =10)
        self.steps = 0
        self.logging_interval = 100
        self.prob = 0.1

    def train(self):
        for epoch in range(0, self.epochs):
            self.train_epoch(epoch)
        self.test_epoch('test')

        return self.model, self.test_loader

    def train_epoch(self,epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.squeeze()
            target = data.squeeze()
            data = data
            rnd = torch.rand_like(data)
            noisy = data
            noisy[rnd<self.prob/2] = 0
            noisy[rnd>(1-self.prob/2)] =1
            data = noisy


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
                rnd = torch.rand_like(data)
                noisy = data
                noisy[rnd<self.prob/2] = 0
                noisy[rnd>(1-self.prob/2)] =1
                data = noisy

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.binary_cross_entropy(output, target.squeeze(), reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True) 
        loss /= 10000*784
        if type_set== 'valid':

            self.writer.add_scalar('Loss/valid',
                     loss,self.steps)

        print('{} set: Average loss: {:.4f}'.format(
            type_set,loss))
        torch.save(self.model.state_dict(),'denae.pth')

        with torch.no_grad():
            for data, target in curr_loader:
                data_o = data.squeeze()
                rnd = torch.rand_like(data_o)
                noisy = data_o
                noisy[rnd<self.prob/2] = 0
                noisy[rnd>(1-self.prob/2)] =1
                data = noisy
                data = data.to(self.device)
                output = self.model(data)
                data2 = data.cpu().detach().numpy()
                output2= output.cpu().detach().numpy()
                plt.imsave(f'.\\Progress5\\{self.hidden_dim}orig.jpeg',data2[0,:,:])
                plt.imsave(f'.\\Progress5\\{self.hidden_dim}back.jpeg',output2[0,:,:])                
                print('val',F.binary_cross_entropy(output.squeeze(), data_o.cuda(), reduction='sum').item()/(256*784))
                if type_set!='valid':
                    print('Testing Denoising',F.binary_cross_entropy(output.squeeze(), data_o.cuda(), reduction='sum').item()/(256*784))
                    stdmodel = stdAE()
                    stdmodel.load_state_dict(torch.load('stdae.pth'))
                    stdmodel.to(self.device)
                    outstd = stdmodel(data)
                    print('Testing Regular',F.binary_cross_entropy(outstd.squeeze(), data_o.cuda(), reduction='sum').item()/(256*784))

                break



if __name__=='__main__':
    train_loader,valid_loader,test_loader,device,size = loadData()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dim',type=int)
#    args = parser.parse_args()
#    c = Neural_Network(train_loader,valid_loader,test_loader,device,size,args.dim)
    for data, target in test_loader:

        data = data.squeeze()
        print(data[0,14,:])

        data_step = data + torch.normal(torch.zeros_like(data),0.8)
        stdmodel = stdAE()
        stdmodel.load_state_dict(torch.load('stdae8.pth'))
        print(device)
        outstd = stdmodel.encode(data)
        outstd_step = outstd+ torch.normal(torch.zeros_like(outstd),0.8)
        manifold_step= stdmodel.decode(outstd_step)
        for i in range(5):
            plt.imsave(f'.\\Progress51\\{i}clean.jpeg',data[i,:,:])
            plt.imsave(f'.\\Progress51\\{i}noise.jpeg',data_step[i,:,:])
            plt.imsave(f'.\\Progress51\\{i}noise_manifold.jpeg',manifold_step.detach().numpy()[i,:,:])                
        break





