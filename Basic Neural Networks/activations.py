import numpy as np
import pandas

def LoadData(type ='normal'):
    train_data = pandas.read_csv('mnist_train.csv')
    train_data = train_data.to_numpy()
    X_train, Y_train = train_data[:,1:],train_data[:,0:1]
    test_data = pandas.read_csv('mnist_test.csv')
    test_data = test_data.to_numpy()
    X_test, Y_test = test_data[:,1:],test_data[:,0:1]
    X_train,X_test = X_train/255,X_test/255
    if type=='noisy':
        X_train = X_train + np.random.normal(0,0.01,X_train.shape)
        X_test = X_test + np.random.normal(0,0.01,X_test.shape)
    return X_train, Y_train, X_test,Y_test



def ReLU(x):
    return np.fmax(0,x)

def ReLUprime(y):
    return np.where(y>0,1,0)


def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
def sigmoidprime(y):
    return  y*(1-y)

def softmax(x):
    x = np.clip(x,-100,100)
    n = np.exp(x - np.expand_dims(np.max(x,axis=1),1 )  )
    return n/np.expand_dims(np.sum(n,axis=1),1)

def tanh(x):
    return np.tanh(x)

def tanhprime(y):
    return 1- y*y


class Nonlinearity():
    def __init__(self,function='ReLU'):
        if function=='ReLU':
            self.function= ReLU
            self.derivative = ReLUprime
        elif function=='sigmoid':
            self.function=sigmoid
            self.derivative=sigmoidprime
        elif function=='tanh':
            self.function=tanh
            self.derivative=tanhprime
        elif function=='softmax':
            self.function=softmax
    def fn(self,x):
        return self.function(x)
    def der(self,y):
        return self.derivative(y)
