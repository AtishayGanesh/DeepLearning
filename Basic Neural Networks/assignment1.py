import numpy as np
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix,f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from activations import *
import cv2
from sklearn.svm import SVC
import sklearn.neighbors
np.seterr(all='raise')
np.set_printoptions(precision=2)


class Linear_layer():
    def __init__(self,input_size=3, layer_size=4,activation='sigmoid',alpha=1e-3, gr= False,fd = False,l2 = 0):
        self.size = layer_size
        self.input_size = input_size
        self.activation_fn = Nonlinearity(activation)
        self.randomsize= (6/(input_size+layer_size))**0.5
        self.weights = np.random.uniform(-self.randomsize,self.randomsize,(self.input_size,self.size))
        self.biases = np.zeros((1,self.size))
        self.lr = alpha
        self.gaussian_reg= gr
        self.gaussian_fwd = fd

        self.l2 = l2


    def forward_pass(self,input):
        if not self.gaussian_fwd:
            return self.activation_fn.fn(input@self.weights + self.biases)
        else:
            noise = np.random.normal(0,0.01,self.biases.shape)
            return self.activation_fn.fn(input@self.weights + self.biases)+ noise

    def backward_pass(self,activations,errors,post_activations):
        assert errors.shape == post_activations.shape
        errors =  errors*self.activation_fn.der(post_activations)
        db = np.expand_dims(np.sum(errors,axis=0),0)/errors.shape[0]
        dw = np.transpose(activations)@errors/errors.shape[0]
        if self.gaussian_reg:
            activations = activations+ np.random.normal(0,0.01,activations.shape)
        assert self.weights.shape ==dw.shape
        assert self.biases.shape==db.shape
        inact_w = (np.sum(np.where(np.absolute(dw)<10**-5,1,0))+np.sum(np.where(np.absolute(db)<10**-5,1,0)))
        sum_w = (np.size(dw)+np.size(db))
        
        self.weights = self.weights*(1-self.l2*self.lr)
        self.biases = self.biases*(1-self.l2*self.lr)
        self.weights = self.weights-self.lr*dw
        self.biases = self.biases -self.lr*db
        return (errors@np.transpose(self.weights)),inact_w,sum_w


class Output_Layer():
    def __init__(self,input_size=3, layer_size=10,activation='softmax',alpha=1e-3,gr = False,fd=False,l2=0):
        self.size = layer_size
        self.input_size =input_size
        self.activation_fn = Nonlinearity(activation)
        self.randomsize= (6/(input_size+layer_size))**0.5
        self.weights = np.random.uniform(-self.randomsize,self.randomsize,(self.input_size,self.size))
        self.biases = np.zeros((1,self.size))
        self.lr = alpha
        self.gaussian_reg= gr
        self.gaussian_fwd = fd
        self.l2=l2


    def forward_pass(self,input):
        if not self.gaussian_fwd:
            return self.activation_fn.fn(input@self.weights + self.biases)
        else:
            noise = np.random.normal(0,0.01,self.biases.shape)
            return self.activation_fn.fn(input@self.weights + self.biases)+ noise
    def backward_pass(self,activations,pred_outputs,targets):
        error_final = pred_outputs - np.squeeze(targets)
        if self.gaussian_reg:
            activations = activations+ np.random.normal(0,0.01,activations.shape)
        db = np.expand_dims(np.sum(error_final,axis=0),0)/error_final.shape[0]
        dw = np.transpose(activations)@error_final/error_final.shape[0]
        inact_w = (np.sum(np.where(np.absolute(dw)<10**-5,1,0))+np.sum(np.where(np.absolute(db)<10**-5,1,0)))
        sum_w = np.size(dw)+np.size(db)
        self.weights = self.weights*(1-self.l2*self.lr)
        self.biases = self.biases*(1-self.l2*self.lr)
        self.weights = self.weights-self.lr*dw
        self.biases = self.biases -self.lr*db
        return (error_final@np.transpose(self.weights)),inact_w,sum_w


def init_NN(activation='sigmoid',lr=1e-3, gr = False,l2=0,fd= False):
    layer_list = [Linear_layer(784,500,activation=activation,alpha=lr,gr=gr,fd=fd),Linear_layer(500,250,activation=activation,alpha=lr,gr=gr,fd=fd),Linear_layer(250,100,activation=activation,alpha=lr,gr=gr,fd=fd),Output_Layer(100,10,alpha=lr,gr=gr,fd=fd)]
    for i in layer_list:
        i.l2 = l2

    return layer_list
def fwd_pass(layer_list,x):
    activation_list = [x]
    for i in layer_list:
        activation_list.append(i.forward_pass(activation_list[-1]))
    return(activation_list)

def evaluate(layer_list,x,y,batchsize=64):
    ypr = []
    for i in range(x.shape[0]//batchsize+1):
        x1 = x[i*batchsize:min(i*batchsize+batchsize,x.shape[0])]
        y1 = y[i*batchsize:min(i*batchsize+batchsize,y.shape[0])]
        activation_list = [x1]
        for i in layer_list:
            activation_list.append(i.forward_pass(activation_list[-1]))
        
        ypr.append(activation_list[-1])
    ypr = np.concatenate(ypr,axis=0)
    return ypr

def train_minibatch(layer_list,x,y_numer):
    y = np.eye(10)[y_numer]
    # fwd pass
    activation_list=fwd_pass(layer_list,x)
    #bwd pass
    errors = []
    sum_w =0
    inact_w = 0
    itr = 0
    acc =  accuracy_score(y_numer,np.argmax(activation_list[-1], axis=1)) 
    loss = log_loss(np.squeeze(y),activation_list[-1])

    for i in layer_list[::-1]:
        if i.__class__.__name__=='Output_Layer':
            error = i.backward_pass(activation_list[itr-2],activation_list[itr-1],y)
            errors.append(error[0])
            inact_w += error[1]
            sum_w +=error[2]
        else:
            error= i.backward_pass(activation_list[itr-3],errors[-1],activation_list[itr-2])
            errors.append(error[0])
            inact_w +=error[1]
            sum_w += error[2]
            itr-=1
    return acc,activation_list,loss,inact_w/sum_w


def trainNN(epochs=15,batchsize=64,activation='sigmoid',lr=1e-3,gr= False,l2=0,data='normal',fd = False):
    ll = init_NN(activation=activation,lr=lr,gr=gr,l2=l2,fd=fd)
    x,y,xt,yt  = LoadData(data)
    sl=0
    train_loss = []
    test_loss = []
    test_i = []
    inact_list= []
    for epoch in range(epochs):
        sl=0

        for i in range(x.shape[0]//batchsize +1):
            x1 = x[i*batchsize:min(i*batchsize+batchsize,x.shape[0])]
            y1 = y[i*batchsize:min(i*batchsize+batchsize,y.shape[0])]
            acc,al,loss,inactp = train_minibatch(ll,x1,y1)
            sl +=acc
            train_loss.append(loss)
            inact_list.append(inactp*100)
            if i%100 ==1:
                test_i.append(epoch*x.shape[0]//batchsize +1+i)
                print(epoch,i,sl/(i+1))
                print(i*batchsize,min(i*batchsize+batchsize,x.shape[0]))
                ypr = evaluate(ll,xt,yt)
                test_loss.append(log_loss(yt,ypr))
                print('test acc:', accuracy_score(np.squeeze(yt),np.argmax(ypr,axis=1)))
    ypr =evaluate(ll,xt,yt)
    print(np.squeeze(yt).shape,np.argmax(ypr,axis=1).shape)
    cnf = confusion_matrix(np.squeeze(yt),np.argmax(ypr,axis=1))
    prec = precision_score(np.squeeze(yt),np.argmax(ypr,axis=1),average='macro')
    rec = recall_score(np.squeeze(yt),np.argmax(ypr,axis=1),average='macro')
    f1 = f1_score(np.squeeze(yt),np.argmax(ypr,axis=1),average='macro')
    return train_loss,test_i,test_loss,(cnf,prec,rec,f1,inact_list)


def main_program(activation='ReLU',lr=1e-3,gr=False,l2=0, data= 'normal',fd=False):
    tr_l,test_i, test_l,metrics = trainNN(epochs=15,batchsize=64,activation='ReLU',lr=lr,gr=gr,l2=l2,data=data,fd= fd)
    plt.plot(tr_l)
    plt.plot(test_i,test_l)
    if data=='noisy':
        str_list = ['{}_noisy_input.png'.format(activation),'inactp{}_noisy_input.png'.format(activation),'stats_noisy_input.txt'.format(activation)]
    elif fd ==True:
        str_list = ['{}_Noise_fwd.png'.format(activation),'inactp{}_Noise_fwd.png'.format(activation),'stats_{}_Noise_fwd.txt'.format(activation)]

    elif not gr and l2==0:
        str_list = ['{}.png'.format(activation),'inactp{}.png'.format(activation),'stats_{}.txt'.format(activation)]

    elif l2==0:
        str_list = ['{}_Noise_bwd.png'.format(activation),'inactp{}_Noise_bwd.png'.format(activation),'stats_{}_Noise_bwd.txt'.format(activation)]
    else:
        str_list = ['{}_L2_Regularization.png'.format(activation),'inactp{}_L2_Regularization.png'.format(activation),'stats_{}_L2_Regularization.txt'.format(activation)]
    plt.title('Training and Test loss over multiple epochs')
    plt.xlabel('mini-batch number')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Test Loss'])

    plt.savefig(str_list[0], bbox_inches='tight')
    plt.clf()
    plt.plot(metrics[4])
    plt.title('Percentage of inactive neurons over epochs')
    plt.xlabel('mini-batch number')
    plt.ylabel('Percentage of inactive neurons')

    plt.savefig(str_list[1], bbox_inches='tight')
    plt.clf()
    print('Confusion Matrix')
    print(metrics[0])
    print('Precision: {}, Recall: {}, F1 score: {}'.format(metrics[1],metrics[2],metrics[3]))
    with open(str_list[2],'w') as fp:
        np.savetxt(fp,metrics[0],fmt='%.5f',delimiter=', ', newline='\n')
        fp.write('\n')
        fp.write('Precision: {}, Recall: {}, F1 score: {}'.format(metrics[1],metrics[2],metrics[3]))


def main():
    for act in [('sigmoid',5e-3),('ReLU',5e-3),('tanh',5e-3)]:

        main_program(act[0],act[1])
    main_program('sigmoid',5e-3,fd =True)

    main_program('sigmoid',5e-3,gr =True)
    main_program('ReLU',5e-3,l2 = 0.001)

    main_program('ReLU',5e-3,data='noisy')



def hog(img):
    winSize = (28,28)
    blockSize = (8,8)
    cellSize = blockSize
    nbins = 9
    blockStride = (4,4)
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 1
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    return hog.compute(img)
    

def using_DL(x,y,xt,yt):
    lr = 5e-3
    activation= 'ReLU'
    input_dim = x.shape[1]
    ll = [Linear_layer(input_dim,32,activation=activation,alpha=lr),Linear_layer(32,16,activation=activation,alpha=lr),Output_Layer(16,10,alpha=lr)]

    #feature extraction
    train_loss = []
    test_loss = []
    test_i = []
    inact_list= []
    epochs=15
    batchsize = 64
    for epoch in range(epochs):
        sl=0

        for i in range(x.shape[0]//batchsize +1):
            x1 = x[i*batchsize:min(i*batchsize+batchsize,x.shape[0])]
            y1 = y[i*batchsize:min(i*batchsize+batchsize,y.shape[0])]
            acc,al,loss,inactp = train_minibatch(ll,x1,y1)
            sl +=acc
            train_loss.append(loss)
            inact_list.append(inactp*100)
            if i%100 ==1:
                test_i.append(epoch*x.shape[0]//batchsize +1+i)
                print(epoch,i,sl/(i+1))
                print(i*batchsize,min(i*batchsize+batchsize,x.shape[0]))
                ypr = evaluate(ll,xt,yt)
                test_loss.append(log_loss(yt,ypr))
                print('test acc:', accuracy_score(np.squeeze(yt),np.argmax(ypr,axis=1)))
    ypr =evaluate(ll,xt,yt)
    print(np.squeeze(yt).shape,np.argmax(ypr,axis=1).shape)
    cnf = confusion_matrix(np.squeeze(yt),np.argmax(ypr,axis=1))
    prec = precision_score(np.squeeze(yt),np.argmax(ypr,axis=1),average='macro')
    rec = recall_score(np.squeeze(yt),np.argmax(ypr,axis=1),average='macro')
    f1 = f1_score(np.squeeze(yt),np.argmax(ypr,axis=1),average='macro')
    metrics = (cnf,prec,rec,f1,inact_list)
    plt.plot(train_loss)
    plt.plot(test_i,test_loss)
    str_list = ['{}_Features.png'.format(activation),'inactp%{}_Features.png'.format(activation),'stats_{}_Features.txt'.format(activation)]
    plt.title('Training and Test loss over multiple epochs')
    plt.xlabel('mini-batch number')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Test Loss'])
    plt.savefig(str_list[0], bbox_inches='tight')
    plt.clf()
    plt.plot(metrics[4])
    plt.title('Percentage of inactive neurons over epochs')
    plt.xlabel('mini-batch number')
    plt.ylabel('Percentage of inactive neurons')
    plt.savefig(str_list[1], bbox_inches='tight')
    plt.clf()
    print('Confusion Matrix')
    print(metrics[0])
    print('Precision: {}, Recall: {}, F1 score: {}'.format(metrics[1],metrics[2],metrics[3]))
    with open(str_list[2],'w') as fp:
        np.savetxt(fp,metrics[0],fmt='%.5f',delimiter=', ', newline='\n')
        fp.write('\n')
        fp.write('Precision: {}, Recall: {}, F1 score: {}'.format(metrics[1],metrics[2],metrics[3]))


    return train_loss,test_i,test_loss,(cnf,prec,rec,f1,inact_list)


def get_features():
    x,y,xt,yt = LoadData()
    print('going to start hog')
    x = x*255
    xt = xt*255
    x = np.reshape(x,(60000,28,28))
    xt = np.reshape(xt,(10000,28,28))
    hogdata = [hog(row.astype('uint8')) for row in x]
    print('converted training data')
    hogdata = np.array(hogdata)
    print(np.squeeze(hogdata).shape)
    trainData = np.float32(hogdata).reshape(60000,-1)
    hogdata_test = [hog(row.astype('uint8')) for row in xt]
    testData = np.float32(hogdata_test).reshape(10000,-1)
    print(trainData.shape)
    responses = y
    print(responses.shape)
    return trainData,y,testData,yt



def using_SVM(x,y,xt,yt):
    clf = SVC(gamma ='auto')
    clf.fit(x,y)
    print("SVM Score is:",clf.score(xt,yt))
def usingKNN(x,y,xt,yt):
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x,y)
    print("KNN Score is",neigh.score(xt,yt))

def feature_extractor():
    x,y,xt,yt = get_features()
    y_f, yt_f = np.squeeze(y.astype('float32')),np.squeeze(yt.astype('float32'))
    #using_DL(x,y,xt,yt)
    using_SVM(x,y_f,xt,yt_f)
    usingKNN(x,y_f,xt,yt_f)

# main()

feature_extractor()