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
def sample(len,number=1):
    inp_o = np.random.randint(0,10,(number,len))
    n_values = 10
    inp = np.eye(n_values)[inp_o]
    target = inp_o[:,1]
    return inp,target

def loadData():
    train = [],[]
    valid =[],[]
    test = [],[]
    device = torch.device("cpu")
    for i in range(3,11):
        a,b,c = sample(i,1000),sample(i,100),sample(i,100)
        train[0].extend(a[0])
        train[1].extend(a[1])
        valid[0].extend(b[0])
        valid[1].extend(b[1])
        test[0].extend(c[0])
        test[1].extend(c[1])
    tensor_trx = ([torch.Tensor([i]) for i in train[0]]) 
    tensor_try = ([torch.Tensor([i]) for i in train[1]])
    tensor_vx = ([torch.Tensor([i]) for i in valid[0]]) 
    tensor_vy = ([torch.Tensor([i]) for i in valid[1]]) 

    tensor_ttx = ([torch.Tensor([i]) for i in test[0]]) 
    tensor_tty = ([torch.Tensor([i]) for i in test[1]]) 

    x,y,z = list(zip(tensor_trx,tensor_try)),list(zip(tensor_vx,tensor_vy)),list(zip(tensor_ttx,tensor_tty))
    shuffle(x)
    shuffle(y)
    shuffle(z)

    tensor_trx,tensor_try = zip(*x)
    tensor_vx,tensor_vy = zip(*y)
    tensor_ttx,tensor_tty = zip(*z)
    return tensor_trx,tensor_try,tensor_ttx,tensor_tty,device,(len(train[0]),len(valid[0]),len(test[0]))

class RNN(nn.Module):
    def __init__(self,latent_dim):
        super(RNN, self).__init__()
        self.hidden_dim = latent_dim
        self.input_dim = 10
        self.num_dir = 1
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                    batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim*self.num_dir,10)
        print('num_dir',self.num_dir)

    def forward(self, x):
        batch_size = x.size(0)
        out, hidden = self.rnn(x)
        out = out[:,-1,:]
        out = self.fc1(out)
        return F.log_softmax(out, dim=1)   


class Neural_Network():
    def __init__(self,latent_dim):
        self.train_x,self.train_y,self.test_x,self.test_y ,self.device,self.size = loadData()
        self.model = RNN(latent_dim)
        self.lr = 5e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=0.0001)
        self.writer = SummaryWriter(log_dir='.\\Q2_runs',flush_secs =10)
        self.steps = 0
        self.logging_interval = 300
        self.epochs =5
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
                target = target.long()
                loss = F.nll_loss(output, target)
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('Loss/train', loss.item(),self.steps)
                self.writer.add_scalar('Accuracy/train', 100*correct / len(target),self.steps)
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
        loss = 0
        with torch.no_grad():
            for data, target in zip(self.test_x,self.test_y):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                target = target.long()
                loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= self.size[-1]
        print(f'Test Set\n Loss:{loss} {correct}/{self.size[-1]} Correctly Identified, Accuracy={correct/self.size[-1]}')
        if epoch ==self.epochs-1:
            for data,target in zip(self.test_x[:5],self.test_y[:5]):
                output = self.model(data)
                target = target.long()
                loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                log_str1 = 'Input Sequence{}'.format(np.argmax(np.squeeze(data.detach().numpy()),axis=1) )

                log_str2 = f'Prediction for 2\'nd element: {pred},\n\
                    Actual 2\'nd element{target}'
                print(log_str1)
                print(log_str2)

                with open('logq2.txt','a') as fp:
                    fp.write(log_str1)
                    fp.write(log_str2)

if __name__=='__main__':
    latent_dim = sys.argv[1]
    print('Training with latent_dim ',latent_dim)
    c = Neural_Network(int(latent_dim))
    c.train()

            
