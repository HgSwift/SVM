import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import random


def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    N = len(y_train)
    alpha = torch.zeros(N)
    alpha.requires_grad = True
    d = x_train.size()[1]
    kernelmat = torch.zeros([N,N])
    for i in range(N):
        for j in range(N):
            kernelmat[i][j] = kernel(x_train[i], x_train[j])
    #w = torch.zeros(d)
    #optimizer = torch.optim.SGD([alpha], lr=lr) 
    for i in range(num_iters):
        loss = (1/2)*torch.sum(torch.outer(alpha*y_train, alpha*y_train)*kernelmat) - alpha.sum()
        print(f"Alpha Grad1: {alpha.grad}")
        #print(loss)
        #optimizer.zero_grad() # zero the gradient buffers
        loss.backward() 
        print(f"Alpha Grad2: {alpha.grad}")
        print(f'Epoch: {i}, Loss: {loss.item()}, alpha: {alpha}')
        #optimizer.step() # does the update
        with torch.no_grad():
            print(f"Alpha Grad3: {alpha.grad}")
            alpha -= lr*alpha.grad
            alpha.clamp_(0, c)
            alpha.grad.zero_()
            print(f"Alpha Grad0: {alpha.grad}")
    return(alpha.detach())
    pass

'''def PIc(alpha):
    alpha2 = torch.sub(torch.transpose(alpha, 0, 1), alpha)
    return(torch.argmin(torch.norm(alpha2, p=2)))
'''

'''def hinge_loss(t):
    loss = 1-t 
    loss[loss < 0] = 0 # PyTorch supports 'numpy-style' indexing
    return loss'''
'''def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1 - y_pred.t() * y_true, min=0))'''

'''def soft_margin_loss_p():
    pred = X_p @ W_p - b_p 
    hinge = hinge_loss(pred*Y_p).mean()
    tikhonov = (W_p**2).sum()
    return C*hinge + tikhonov'''

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    N = len(y_train)
    d = x_train.size()[1]
    M = x_test.size()[0]
    out = torch.zeros(M)
    ker = torch.zeros(N)
    for i in range(M):
        for j in range(N):
            ker[j] = alpha[j]*y_train[j]*kernel(x_train[j], x_test[i])
        out[i] = ker.sum()
    return(out)
    pass

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!
        
        self.conv18 = nn.Conv2d(1, 8, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv84 = nn.Conv2d(8, 4, kernel_size=3)
        self.linear = nn.Linear(4, 10)
        #self.relu = F.relu
        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb = xb.unsqueeze(1)
        #print(xb.size())
        N = xb.size()[0]
        y = F.relu(self.conv18(xb))
        y = F.relu(self.maxpool(y))
        y = F.relu(self.conv84(y))
        y = y.view(N,4)
        #print(y.size())
        y = self.linear(y)
        return y
        #return(model_1(xb))
        pass

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    gamma = 1.0
    if(batch_size == 2):
        batch_size = 1
        gamma = 0.95
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)
    train_losses = []
    test_losses = []
    #loss = hw2_utils.epoch_loss(net, loss_func, train_dl)
    #print(hw2_utils.epoch_loss(net, loss_func, train_dl)) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    with torch.no_grad():
        train_losses.append(hw2_utils.epoch_loss(net, loss_func, train_dl))
        test_losses.append(hw2_utils.epoch_loss(net, loss_func, train_dl))
    for i in range(n_epochs): #training
        for i, (X_batch, Y_batch) in enumerate(train_dl): 
            hw2_utils.train_batch(net, loss_func, X_batch, Y_batch, optimizer)
        with torch.no_grad():
            train_losses.append(hw2_utils.epoch_loss(net, loss_func, train_dl))
            test_losses.append(hw2_utils.epoch_loss(net, loss_func, test_dl))
        scheduler.step()
    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    return train_losses, test_losses

'''train, test = hw2_utils.torch_digits()
net1 = DigitsConvNet()
optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.005)
net2 = DigitsConvNet()
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.005)
net3 = DigitsConvNet()
optimizer3 = torch.optim.SGD(net3.parameters(), lr=0.005)
train1, test1 = fit_and_evaluate(net1, optimizer1, torch.nn.CrossEntropyLoss(), train, test, 30, 1)
train2, test2 = fit_and_evaluate(net2, optimizer2, torch.nn.CrossEntropyLoss(), train, test, 30, 2)
train3, test3 = fit_and_evaluate(net3, optimizer3, torch.nn.CrossEntropyLoss(), train, test, 30, 16)
#print(train1)
plt.plot(range(len(train1)), train1, c='#ff0000', marker='.')
plt.plot(range(len(train2)), train2, c='#00ff00', marker='.')
plt.plot(range(len(train3)), train3, c='#0000ff', marker='.')
plt.plot(range(len(test1)), test1, c='#ff8888', marker='.')
plt.plot(range(len(test2)), test2, c='#88ff88', marker='.')
plt.plot(range(len(test3)), test3, c='#8888ff', marker='.')
plt.show()'''




x, y = hw2_utils.xor_data()
def predictor(x,y):
    ker = hw2_utils.rbf(4)
    alpha = svm_solver(x, y, 0.1, 100, kernel=ker)
    return lambda x_test: svm_predictor(alpha, x, y, x_test, kernel=ker)
pred_fxn = predictor(x,y)
hw2_utils.svm_contour(pred_fxn)