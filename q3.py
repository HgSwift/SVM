import torch
from torch.functional import Tensor

x = torch.tensor([[1],[1],[0],[0],[1],[1]])
y = torch.tensor([[0],[1],[0]])

a = torch.tensor([
    [1.0, 2.0, -3.0, 0.0, 1.0, -3.0],
[3.0, 1.0, 2.0, 1.0, 0.0, 2.0],
[2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
[1.0, 0.0, 2.0, 1.0, -2.0, 2.0]])
b = torch.tensor([
    [1, 2, -2, 1], 
    [1, -1, 1, 2], 
    [3, 1, -1, 1]])

def calcA(j, a, b, x):
    j-=1
    out = 1
    for i in range(6):
        out += a[j][i]*x[i][0]
    return(out)

def calcZ(j, a, b, x):
    
    return(1/(1+torch.exp(-1*calcA(j,a,b,x))))

def calcB(k, a, b, x):
    out = 1
    for j in range(4):
        out+=(b[k][j]*calcZ(j, a, b, x))
    return(out)

def yhat(k,a,b,x):
    bot = 0
    top = torch.exp(calcB(k, a, b, x))
    for l in range(3):
        bot += torch.exp(calcB(l,a,b,x))
    return(top/bot) 

def loss(a, b, x, y):
    out = 0
    for k in range(3):
        out += y[k][0]*torch.log(yhat(k,a,b,x))

print(calcA(1,a,b,x))
print(calcZ(1,a,b,x))
print(calcA(3,a,b,x))
print(calcZ(3,a,b,x))
print(calcB(1,a,b,x))
print(yhat(1,a,b,x))
print(loss(a, b, x, y))


