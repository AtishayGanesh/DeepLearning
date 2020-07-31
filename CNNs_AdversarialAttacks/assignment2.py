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


class CNN(nn.Module):
    def __init__(self,batchnorm=False):
        super(CNN, self).__init__()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.conv2_bn = nn.BatchNorm2d(32)
            self.fc1_bn = nn.BatchNorm1d(500)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels =32,kernel_size= 3, stride = 1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,kernel_size= 3, stride = 1,padding=1)
        self.fc1 = nn.Linear(4*4*98, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*98)
        x = (self.fc1(x))

        if self.batchnorm:
            x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def forward_logits(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*98)
        x = (self.fc1(x))

        if self.batchnorm:
            x = self.fc1_bn(x)
        x = F.relu(x)
        return self.fc2(x)


def loadData(batch_size=64):
    cuda_train = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_train else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_train else {}
    train = np.arange(0,54000,1,dtype=int)
    val = np.arange(54000,60000,1,dtype=int)
    train_sampler = SubsetRandomSampler(train)
    valid_sampler = SubsetRandomSampler(val)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),batch_size= batch_size, shuffle=False,sampler=train_sampler,**kwargs)
    
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),batch_size= batch_size, shuffle=False,sampler=valid_sampler,**kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batch_size, shuffle=True,**kwargs)
    return train_loader,valid_loader,test_loader,device,(len(train_sampler),len(valid_sampler),len(test_loader.dataset))

class Neural_network():
    def __init__(self,lr=5e-3,momentum=0.5,epochs=10,logging_interval=100,batchnorm=False):
        self.lr =lr
        self.momentum = momentum
        self.epochs = epochs
        self.steps = 0
        self.logging_interval = logging_interval
        self.train_loader,self.valid_loader,self.test_loader,self.device,self.size = loadData()
        self.writer = SummaryWriter(flush_secs =10)
        self.model = CNN(batchnorm).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

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

def plot_layers(name,self,input,output):
    on = output.detach().numpy()
    print(on.shape)
    for i in range(5):
        plt.figure(i)
        plt.imshow(np.squeeze(on)[i])
        plt.imsave(f'Part_21_{i}_{name}.png',np.squeeze(on)[i])
    plt.show()

def occlusions(x):
    x1 = []
    for i in range(0,24):
        for j in range(0,24):
            x_c =x.copy()
            x_c[i:i+5,j:j+5] =0
            x1.append(x_c)
    return(np.expand_dims(np.array(x1),axis=1))



def targeted_attack(start,target,model):
    train_loader,valid_loader,test_loader,device,size = loadData(batch_size=10)
    while True:
        try:
            inputs, classes = next(iter(test_loader))
            classes = classes.numpy()
            aw =np.argwhere(classes==start)
            print(classes[aw[0]],aw[0])
            start_image = inputs[aw[0]]
            break
        except IndexError:
            pass
    while True:
        try:
            inputs, classes = next(iter(test_loader))
            classes = classes.numpy()
            aw =np.argwhere(classes==target)
            print(classes[aw[0]],aw[0])
            target_image = inputs[aw[0]]
            break
        except IndexError:
            pass

    step = 0.3
    p = 0
    beta = 0.0001
    X1 = torch.tensor(start_image, requires_grad=True).type('torch.FloatTensor')
    device = torch.device( "cpu")

    while(p<0.90):
        X1 = X1.to(device)
        loss = model.forward_logits(X1)[0][target] - beta*F.mse_loss(start_image,target_image)
        prob = model(X1)
        p = np.exp(prob.detach().numpy())[0,target]
        g = torch.autograd.grad(loss,X1)
        X1 = X1+step*g[0]
        print(p)
    print(np.exp(prob.detach()))
    plt.imshow(np.squeeze(X1.detach().numpy()))
    plt.colorbar()
    plt.savefig(f'.\\Q3T\\Part_31_{start}to{target}_99.png')
    plt.clf()


def addingNoise(start,target,model):
    train_loader,valid_loader,test_loader,device,size = loadData(batch_size=32)
    while True:
        try:
            inputs, classes = next(iter(test_loader))
            classes = classes.numpy()
            aw =np.argwhere(classes==start)
            print(classes[aw[0]],aw[0])
            start_image = inputs[aw[0]]
            break
        except IndexError:
            pass
    while True:
        try:
            inputs, classes = next(iter(test_loader))
            classes = classes.numpy()
            aw =np.argwhere(classes==target)
            print(classes[aw[0]],aw[0])
            target_image = inputs[aw[0]]
            break
        except IndexError:
            pass

    step = 0.2
    p = 0
    X1 = torch.tensor(start_image, requires_grad=True).type('torch.FloatTensor')
    X3 = torch.tensor(np.zeros((1,1,28,28)),requires_grad=True).type('torch.FloatTensor')
    X2 = X1+X3
    device = torch.device( "cpu")

    while(p<0.99):
        X2 = X2.to(device)
        loss = model.forward_logits(X2)[0][target]
        prob = model(X2)
        p = np.exp(prob.detach().numpy())[0,target]
        g = torch.autograd.grad(loss,X3)
        X3 = X3+step*g[0]
        X2 = X1 +X3
        print(p)

    
    print(np.exp(prob.detach()))
    f, axarr = plt.subplots(2,1)
    im2 = axarr[0].imshow(np.squeeze(X2.detach().numpy()))
    f.colorbar(im2,ax = axarr[0])
    axarr[0].set_title('Adversial Image')
    im1 = axarr[1].imshow(np.squeeze(X3.detach().numpy()))
    axarr[1].set_title('Noise',y=0.96)
    f.colorbar(im1,ax = axarr[1])

    plt.savefig(f'.\\Q3Noise\\Part_32_{start}to{target}_99.png')
    plt.clf()
    return X3
    




if __name__=="__main__":
    
    if len(sys.argv) == 1:
        c = Neural_network()
        model,test_loader = c.train()

        #torch.save(model.state_dict(),"mnist_cnn.pt")
    elif sys.argv[1] =='Part_11':
        kwargs= {'batchnorm':'True'}
        c = Neural_network(**kwargs)
        model,test_loader = c.train()

        torch.save(model.state_dict(),"mnist_cnn_bn.pt")


    elif sys.argv[1] =='Part_12':
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.eval()
        train_loader,valid_loader,test_loader,device,size = loadData()
        inputs, classes = next(iter(test_loader)) 
        for i in range(5):
            x = inputs.numpy()[i]
            plt.figure(i)
            #plt.imshow(np.squeeze(x))
            plt.imsave(f'Part_2_{i}.png',np.squeeze(x))

        #plt.show()
        pred = model(inputs)
        log_str = 'Predicted Class {}\nActual Class {}'.format(np.argmax(pred.detach().numpy(),axis=1)[:5],classes.numpy()[:5])
        print(log_str)
        with open('log.txt','a') as fp:
            fp.write(log_str)

    elif sys.argv[1]=='Part_20':
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        on = model.conv1.weight.data.numpy()
        model.eval()
        print(on.shape)
        for i in range(5):
            plt.figure(i)
            plt.imshow(np.squeeze(on)[i])
            plt.colorbar()
            plt.savefig(f'Part_20_{i}_conv1.png')
        plt.show()
        on = model.conv2.weight.data.numpy()
        print(on.shape)
        for i in range(5):
            plt.figure(i)
            plt.imshow(np.squeeze(on)[0][i])
            plt.colorbar()
            plt.savefig(f'Part_20_{i}_conv2.png')
        plt.show()

    elif sys.argv[1]=='Part_21':
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.eval()
        train_loader,valid_loader,test_loader,device,size = loadData(batch_size=1)
        inputs, classes = next(iter(test_loader))
        model.conv1.register_forward_hook(partial(plot_layers, 'conv1'))
        model.conv2.register_forward_hook(partial(plot_layers, 'conv2'))
        pred = model(inputs)
        x = inputs.numpy()[0]
        plt.figure(0)
        plt.imshow(np.squeeze(x))
        plt.imsave(f'Part_21_input.png',np.squeeze(x))
        plt.show()

    elif sys.argv[1]=='Part_22':
        '''Occluding parts of the image'''
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.eval()
        train_loader,valid_loader,test_loader,device,size = loadData(batch_size=10)
        inputs, classes = next(iter(test_loader))
        classes = classes.numpy()
        print(classes)
        for i in range(10):
            x = inputs.numpy()[i]
            x1 = occlusions(np.squeeze(x))
            pred = model(torch.from_numpy(x1))

            pred = pred.detach().numpy()
            lin_graph = pred[:,classes[i]]
            lin_graph = np.reshape(lin_graph,(24,24))
            plt.clf()
            f, axarr = plt.subplots(2,1)
            im2 = axarr[0].imshow(np.squeeze(x))
            im1 = axarr[1].imshow(np.exp(lin_graph))
            f.colorbar(im1)
            plt.savefig(f'Part_22_{i}.png')


    elif sys.argv[1] == 'Part_30':
        '''Non Targeted Attack'''
        X = np.random.normal(0,0.1,(1,1,28,28))
        X = X.astype('d')
        X = np.clip(X,-1,1)
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.train()
        step = 0.1
        for i in range(10):
            p = 0
            print(X.shape)
            z = np.array([i])
            target =torch.from_numpy(z).type('torch.LongTensor')

            X1 = torch.tensor(X, requires_grad=True).type('torch.FloatTensor')
            device = torch.device( "cpu")
            plist= []
            while(p<0.99):
                X1, target = X1.to(device), target.to(device)
                loss = model.forward_logits(X1)
                prob = model(X1)
                p = np.exp(prob.detach().numpy())[0,i]
                g = torch.autograd.grad(loss[0][i],X1)
                X1 = X1+step*g[0]
                print(p)
                plist.append(loss[0][i].detach().numpy())
            plt.imshow(np.squeeze(X1.detach().numpy()))
            plt.colorbar()
            plt.title(f'Non Targeted Attack with Accuracy=0.99 for class {i}')
            plt.savefig(f'Part_30_{i}_99.png')
            plt.clf()
            plt.plot(plist)
            plt.title(f'Cost Function for class {i}')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.savefig(f'Part_30_{i}_cost.png')
            plt.clf()


    elif sys.argv[1] == 'Part_31':
        try:
            start,target = int(sys.argv[2]),int(sys.argv[3])
        except:
            raise ValueError('Not enough parameters')
        if start not in range(0,10) or target not in range(0,10):
            raise ValueError('Wrong Input')
        '''Targeted Attack'''
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.train()


        targeted_attack(start,target,model)

    elif sys.argv[1] =='Part_31_All':
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.train()
        for start in range(0,10):
            for target in range(0,10):
                if start != target:
                    targeted_attack(start,target,model)


    elif sys.argv[1] == 'Part_32':
        try:
            start,target = int(sys.argv[2]),int(sys.argv[3])
        except:
            raise ValueError('Not enough parameters')
        if start not in range(0,10) or target not in range(0,10):
            raise ValueError('Wrong Input')
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.train()
        addingNoise(start,target,model)

    elif sys.argv[1] == 'Part_32_All':
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.train()
        noise_layer = []
        for start in range(0,10):
            noise_layer.append( addingNoise(start,(start+1)%10,model))
        train_loader,valid_loader,test_loader,device,size = loadData(batch_size=10)
        inputs, classes = next(iter(test_loader))
        model.eval()
        pred =[[] for j in range(10) ]
        for j in range(10):
            for noisemat in noise_layer:
                print(inputs[j:j+1].shape, noisemat.shape)
                pred_prob = model(inputs[j:j+1] +noisemat)
                pred[j].append( np.argmax(pred_prob.detach().numpy()))
        true = classes.numpy()
        actual =np.array(pred).T 
        attempted = np.mod(np.arange(10) +1,10)
        print(attempted) 
        print(true)
        print(actual)

