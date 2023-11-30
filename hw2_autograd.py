import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader



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
    if c == None:
        c = 0
    N = len(y_train)
    alpha = torch.zeros(N)
    dim2 = x_train.size()[1]
    dataloader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=1) # create dataloader
    model = torch.nn.Linear(dim2,1) # create linear model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # create SGD optimizer
    print(x_train)
    print(y_train)
    for i in range(num_iters):
        #print(i)
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad() # zero the gradient buffers
            pred = model(X_batch)
            loss = c*hinge_loss(pred*Y_batch).mean() + (model.weight**2).sum()
            loss.backward() 
            optimizer.step() # does the update 
        #print(f'Epoch: {i}, Loss: {loss.item()}')
    return(model.parameters.detach())
    pass

def PIc(alpha):
    alpha2 = torch.sub(torch.transpose(alpha, 0, 1), alpha)
    return(torch.argmin(torch.norm(alpha2, p=2)))
    
def hinge_loss(t):
    loss = 1-t 
    loss[loss < 0] = 0 # PyTorch supports 'numpy-style' indexing
    return loss
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
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss

    return train_losses, test_losses












X = torch.tensor([[ 1.5000,  1.0656,  0.0118,  1.2507,  1.0893],
        [ 1.5000, -0.3135,  0.7670, -0.0531,  1.1016],
        [ 1.5000,  1.0655, -1.9451,  0.3395,  0.6804],
        [ 1.5000, -0.1774,  1.3949,  1.4096,  0.9260],
        [ 1.5000,  0.3292, -0.3564, -0.1576,  0.2959],
        [ 1.5000, -0.1191, -0.1317, -0.0774,  1.0874],
        [ 1.5000,  0.5885, -0.3708,  1.8073,  0.0215],
        [ 1.5000,  0.2781,  0.0723,  1.5395, -1.0168],
        [ 1.5000,  1.2265,  1.8873, -0.1213,  1.5785],
        [ 1.5000,  1.6847,  1.3043,  0.1651,  0.4341],
        [-0.5000, -0.3531,  1.7268, -1.5151,  0.6955],
        [-0.5000, -0.0662,  0.9007, -0.2016,  0.6341],
        [-0.5000,  1.5825, -1.0422,  1.1945,  0.7914],
        [-0.5000,  1.6093,  0.3770, -0.1921,  1.7534],
        [-0.5000, -0.3658, -0.6796, -0.5491, -0.5066],
        [-0.5000,  0.6088, -0.5973,  2.4052,  0.8173],
        [-0.5000,  1.4211,  0.2432,  0.1424, -0.5653],
        [-0.5000,  0.5840, -0.2976,  0.5422, -0.0356],
        [-0.5000, -1.8104,  1.9792, -0.5326,  1.6311],
        [-0.5000,  0.9561,  1.4089,  0.7372, -0.6251]])
Y = torch.tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1.])

print(svm_solver(X, Y, 0.1, 100, kernel=hw2_utils.poly(degree=1), c=0))