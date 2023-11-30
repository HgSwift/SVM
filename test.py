import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
C = 1.0
X, Y = make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)
Y[Y == 0] = -1 # to have +/- 1 labels
Y_2d = Y[:,np.newaxis].astype('f') 
print(f' Y shape: {Y.shape}, Y_2d shape: {Y_2d.shape}')
X = X.astype('f') 
X_p = torch.tensor(X) 
Y_p = torch.tensor(Y_2d)
W = np.random.rand(2,1).astype('f') 
b = np.random.rand(1,1).astype('f') 

W_p = torch.tensor(W, requires_grad=True) 
b_p = torch.tensor(b, requires_grad=True)
def hinge_loss(t):
    loss = 1-t 
    loss[loss < 0] = 0 # PyTorch supports 'numpy-style' indexing
    return loss

def soft_margin_loss_p():
    pred = X_p @ W_p - b_p 
    hinge = hinge_loss(pred*Y_p).mean()
    tikhonov = (W_p**2).sum()
    return C*hinge + tikhonov
def gradient_step_p(lr=0.1):
    loss = soft_margin_loss_p()
    loss.backward() 
    W_p.data -= lr*W_p.grad
    b_p.data -= lr*b_p.grad
    W_p.grad.zero_() # set gradients to zero (otherwise next gradients will be added)
    b_p.grad.zero_()
    return loss.item()



W_p_final = W_p.detach().numpy() # detach it from the computational graph
b_p_final = b_p.detach().numpy()

print('PyTorch W:', W_p_final, 'PyTorch b:', b_p_final, sep='\n')


from torch.utils.data import TensorDataset, DataLoader

dataloader = DataLoader(TensorDataset(X_p, Y_p), shuffle=True, batch_size=10) # create dataloader
model = torch.nn.Linear(2,1) # create linear model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # create SGD optimizer

print(f' W: {model.weight}')

def train(epochs=6):
    for i in range(epochs):
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad() # zero the gradient buffers
            pred = model(X_batch)
            loss = C*hinge_loss(pred*Y_batch).mean() + (model.weight**2).sum()
            loss.backward() 
            optimizer.step() # does the update 
        print(f'Epoch: {i}, Loss: {loss.item()}')

train()
print(model.weight.t())