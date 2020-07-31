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


class RNN(nn.Module):
    def __init__(self,device,bidir=False,lstm=False):
        super(RNN, self).__init__()
        self.hidden_dim =128
        self.input_dim = 28
        self.device = device
        self.bidir = bidir
        self.lstm = lstm
        self.num_dir = 2 if bidir else 1
        if lstm:
            self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                    batch_first=True,bidirectional=self.bidir)

        else:

            self.rnn = nn.RNN(input_size=self.input_dim,hidden_size=self.hidden_dim,
                    batch_first=True,bidirectional=self.bidir)
        self.fc1 = nn.Linear(self.hidden_dim*self.num_dir,10)
        self.ct = 0
        print('num_dir',self.num_dir)

    def forward(self, x):
        batch_size = x.size(0)
        out, hidden = self.rnn(torch.squeeze(x))
        out = out[:,-1,:]
        self.ct +=1
        out = self.fc1(out)
        return F.log_softmax(out, dim=1)   


class Neural_network():
    def __init__(self,bidir=False,lr=1e-3,epochs=10,logging_interval=25,lstm=False):
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = loadData()
        self.model = RNN(self.device,bidir,lstm)
        self.model.to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=0.01)
        self.writer = SummaryWriter(flush_secs =10)
        self.steps = 0
        self.logging_interval = logging_interval
        self.bidir = bidir
        self.lstm = lstm
        self.tag_str = 'lstm' if lstm else 'rnn'
        self.tag_str += 'bidir' if bidir else ''



    def train(self):
        for epoch in range(0, self.epochs):
            self.train_epoch(epoch)
        self.test_epoch('test')

        return self.model, self.test_loader

    def train_epoch(self,epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            correct = 0
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('Loss/train', loss.item(),self.steps)
            self.writer.add_scalar('Accuracy/train', 100*correct / len(target),self.steps)
            self.steps +=1

            if batch_idx % self.logging_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f})%]\tLoss: {:.6f}\n'.format(
                    epoch+1, batch_idx * len(data), self.size[0],
                    100.*batch_idx*len(data) /self.size[0], loss.item()),self.steps)
                self.test_epoch('valid')
                self.model.train()

    def test_epoch(self,type_set):
        self.model.eval()
        curr_loader = self.valid_loader if type_set == 'valid' else self.test_loader
        size_value = 1 if type_set =='valid' else 2
        correct = 0
        loss = 0
        with torch.no_grad():
            for data, target in curr_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= self.size[size_value]
        if type_set== 'valid':

            self.writer.add_scalar('Loss/valid',
                     loss,self.steps)
            self.writer.add_scalar('Accuracy/valid',
                      100.*correct / self.size[size_value],self.steps)

        print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            type_set,loss, correct, self.size[size_value],
            100.*correct/self.size[size_value]),self.steps)
        if type_set !='valid':
            with open('log.txt','a') as fp:
                fp.write(f'{self.tag_str}')
                fp.write(f'Test Set Accuracy:{100.*correct/self.size[size_value]}%')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bidir',action='store_true')
    parser.add_argument('--lstm',action='store_true')
    args = parser.parse_args()
    c = Neural_network(bidir = args.bidir,lstm= args.lstm)
    model,test_loader = c.train()
    inputs, classes = next(iter(test_loader))
    print(c.device)
    inputs.to(c.device)
    for i in range(5):
        x = inputs.numpy()[i]
        plt.figure(i)
        #plt.imshow(np.squeeze(x))
        tag_str = 'lstm' if args.lstm else 'rnn'
        tag_str +='_bidir' if args.bidir else ''
        
        plt.imsave(f'{tag_str}_{i}.png',np.squeeze(x))

    #plt.show()
    pred = model(inputs)
    log_str = 'Predicted Class {}\nActual Class {}'.format(np.argmax(pred.detach().numpy(),axis=1)[:5],classes.numpy()[:5])
    print(log_str)
    with open('log.txt','a') as fp:
        fp.write(log_str)

