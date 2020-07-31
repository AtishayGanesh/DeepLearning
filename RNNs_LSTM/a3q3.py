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
from torch.utils.data import TensorDataset,DataLoader
from random import shuffle
import argparse
import json


def sample(types,args,lu=None):
    if types==1:
        lower = 2
        upper = 2**20
    else:
        lower =2**args.fix
        upper =2**(args.fix+1)
    if lu:
        lower = 2**lu
        upper = 2**(lu+1)

    i1 = int(np.random.randint(lower,upper,1))
    i2 = int(np.random.randint(lower,upper,1))
    inp_1 = [int(x) for x in bin(i1)[2:]]
    inp_2 = [int(x) for x in bin(i2)[2:]]
    o_p = [int(x) for x in bin(i2+i1)[2:]][::-1]
    l = len(o_p)
    inp_1 = np.concatenate((np.array(inp_1[::-1]),np.zeros(l-len(inp_1))))
    inp_2 = np.concatenate((np.array(inp_2[::-1]),np.zeros(l-len(inp_2))))
    inp = np.stack((inp_1,inp_2),axis=-1)
    op = np.expand_dims(np.array(o_p),-1)
    return inp,op

def loadData(type_data):
    train = [],[]
    device = torch.device("cpu")
    if args.fix:
        types = 2,1
    else:
        types = 1,1
    for i in range(2000):
        a = sample(types[0],args)
        train[0].append(a[0])
        train[1].append(a[1])
    for j in range(2,21):
        for i in range(200):
            a = sample(types[1],args,lu=j)
            train[0].append(a[0])
            train[1].append(a[1])

    tensor_trx = ([torch.Tensor([i]) for i in train[0]]) 
    tensor_try = ([torch.Tensor([i]) for i in train[1]])
    print(tensor_trx[0].shape)
    print(tensor_try[0].shape)
    return tensor_trx[0:2000],tensor_try[0:2000],tensor_trx[2000:],tensor_try[2000:],device,(2000,200)


class RNN(nn.Module):
    def __init__(self,args):
        super(RNN, self).__init__()
        self.hidden_dim = args.memory
        self.input_dim = 2
        self.num_dir = 1
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                    batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim*self.num_dir,1)
        print('num_dir',self.num_dir)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc1(out)
        return torch.sigmoid(out) 


class Neural_Network():
    def __init__(self,args):
        self.args = args
        self.train_x,self.train_y,self.test_x,self.test_y ,self.device,self.size = loadData(args)
        self.model = RNN(args)
        self.lr = 1e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=1e-3)
        self.writer = SummaryWriter(log_dir='.\\Q3_runs',flush_secs =10)
        self.steps = 0
        self.logging_interval = 300
        self.epochs =7
        self.loss = nn.BCELoss(reduction='sum') if args.celoss else nn.MSELoss(reduction='sum')
        print('hi')
    def train(self):
        for epoch in range(self.epochs):
            avg_loss = 0
            ct= 0
            self.model.train()
            for index,(x,y) in enumerate(zip(self.train_x,self.train_y)):
                data,target = x.to(self.device),y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                correct = 0
                total = 0
                target = target
                loss = F.mse_loss(output, target,reduction='sum')
                pred = output

                a = np.equal(np.around(pred.detach().numpy()),target.detach().numpy())
                correct += np.sum(a)
                total +=len(pred[0])
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('Loss/train', loss.item(),self.steps)
                self.writer.add_scalar('Accuracy/train', 100*correct /total,self.steps)
                self.steps +=1
                avg_loss +=loss.item()
                ct +=1
                if index % self.logging_interval == 0:
                    print(f'epoch: {epoch}, index {index}')
                    print(avg_loss/ct)
                    avg_loss=0
                    ct =0

            self.test_epoch(epoch)

    def test_epoch(self,epoch):
        self.model.eval()
        correct = 0
        total= 0
        loss = 0
        with torch.no_grad():
            li = [0 for i in range(2,21)]
            for data, target in zip(self.test_x,self.test_y):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                target = target
                loss += self.loss(output, target).item()
                pred = output
                a = np.equal(np.around(pred.detach().numpy()),target.detach().numpy())
                correct += np.sum(a)
                total +=len(pred[0])
                li[len(pred[0])-4] +=correct/(total*200)
        print(li)
        loss /= self.size[-1]
        print(f'Test Set\n Loss:{loss} {correct}/{total} Correctly Identified, Accuracy={correct/total}')

        if epoch ==self.epochs-1:
            for data,target in zip(self.test_x[-2:],self.test_y[-2:]):
                output = self.model(data)
                target = target
                loss += self.loss(output, target).item()
                pred = output
                log_str1 = 'Input Sequence{}\n'.format(data.detach().numpy())

                log_str2 = f'Prediction bitwise: {np.round(pred.detach().numpy())},\n\
                    Actual bitwise: {target.detach().numpy()}\n'
                print(log_str1)
                print(log_str2)

                with open('logq3.txt','a') as fp:
                    if args.fix == None:
                        args.fix = -1 
                    fp.write(str(args))
                    fp.write(log_str1)
                    fp.write(log_str2)
            with open('logq3_basic.txt','a') as fp:

                fp.write(str(args)+'\n')
                fp.write(str(li)+'\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('memory',type=int)
    parser.add_argument('--fix',type=int)
    parser.add_argument('--celoss',action='store_true')
    args = parser.parse_args()
    print(str(args))
    c = Neural_Network(args)
    c.train()

            
