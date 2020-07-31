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
import argparse
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor

def Gaussian(x,mu=2,sigma=0.2):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2)/2)
class Generator(nn.Module):
    def __init__(self,device):
        self.size = 8
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(1,self.size)
        self.fc2 = nn.Linear(self.size,self.size)
        self.fc3 = nn.Linear(self.size,1)
        self.ct = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out   

class Discriminator(nn.Module):
    def __init__(self,device):
        super(Discriminator, self).__init__()
        self.size = 8
        self.fc1 = nn.Linear(1,self.size)
        self.fc2 = nn.Linear(self.size,self.size)
        self.fc3 = nn.Linear(self.size,1)
        self.ct = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return torch.sigmoid(out)    




class Neural_network():
    def __init__(self,bidir=False,lr=1e-3,epochs=60,logging_interval=25,lstm=False):
        #self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = loadData()
        self.device = torch.device("cuda" )

        self.Generator = Generator(self.device)
        self.Generator.to(self.device)
        self.Discriminator = Discriminator(self.device)
        self.Discriminator.to(self.device)

        self.epochs = epochs
        self.glr = lr
        self.dlr = 10*lr
        self.optimizerg = torch.optim.Adam(self.Generator.parameters(), lr=self.glr,weight_decay=0.001)
        self.optimizerd = torch.optim.Adam(self.Discriminator.parameters(), lr=self.dlr,weight_decay=0.001)

        self.loss = torch.nn.BCELoss()



    def train(self):
        for epoch in range(0, self.epochs):
            self.train_epoch(epoch)
        #self.test_epoch()

        return self.Generator

    def train_epoch(self,epoch):
        self.Generator.train()
        self.Discriminator.train()
        for batch_idx in range(100):
            data = torch.randn(64,1).cuda()*0.2+2
            noise = torch.rand(64,1).cuda()*2-1
            valid = Variable(Tensor(64, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(64, 1).fill_(0.0), requires_grad=False)
            data, valid = data.to(self.device), valid.to(self.device)
            noise, fake = noise.to(self.device), fake.to(self.device)
            
            #training generator
            self.optimizerg.zero_grad()
            gen_data = self.Generator(noise)
            if batch_idx%10==0:
                g_loss = self.loss(self.Discriminator(gen_data),valid)
                g_loss.backward()
                self.optimizerg.step()

            #training the discriminator
            self.optimizerd.zero_grad()
            fake_loss = self.loss(self.Discriminator(gen_data.detach()),fake)
            real_loss = self.loss(self.Discriminator(data),valid)
            d_loss = 0.5*(real_loss+fake_loss)
            d_loss.backward()
            self.optimizerd.step()
            if batch_idx%50==0:
                print("Epoch:{} {} D loss: {} G loss: {}".format(epoch,batch_idx, d_loss.item(), g_loss.item()))
                self.test_epoch(epoch,batch_idx)



    def test_epoch(self,epoch,batch_idx):
        self.Generator.eval()
        self.Discriminator.eval()
        plt.ion()
        test_noise = (2*torch.rand(1000,1)-1).cuda()
        plt.figure(0)
        plt.hist(test_noise.cpu().detach().numpy(),20,density=True)
        x = np.linspace(-3,3,100,dtype='d')
        x1 = np.expand_dims(x,-1)
        plt.plot(x,Gaussian(x))
        gen_out =self.Generator(test_noise).cpu().detach().numpy()
        plt.hist(gen_out,density=True)
        plt.plot(x,Gaussian(x,mu=np.mean(gen_out),sigma=np.std(gen_out)))
        plt.plot(x,self.Discriminator(torch.from_numpy(x1).type(torch.FloatTensor).cuda()).cpu().detach().numpy() )
        plt.xlim(-3,3)
        plt.ylim(0,5)
        plt.legend(('Data_dist','Gen_dist','Discriminator','Noise','Gen_hist'))
        plt.savefig(f'.\\GAN\\Epoch{epoch}batch{batch_idx}')
        plt.show()
        plt.pause(0.02)
        plt.clf()


if __name__=='__main__':
    c = Neural_network()
    model = c.train()

# Final Loss Epoch:60 0 D loss: 0.6824350357055664 G loss: 0.7154850363731384