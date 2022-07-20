import numpy as np
import torch

###Inputs
W = np.array([1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0])
W = torch.tensor(W.reshape(3,3), requires_grad=True)
x = np.array([2.0,1.0,0.0])
x = torch.tensor(x.reshape(3,1), requires_grad=True)
yhat = np.array([1.76159,1.46212,1.76159])
yhat = torch.tensor(yhat.reshape(3,1), requires_grad=True)
sig = torch.nn.Sigmoid()

###L2 loss
def MSE (yhat, y):
    #print(y.size())
    #print(len(y))
    r = torch.sub(yhat,y)
    #print(r)
    r1 = torch.square(r)
    r2 = torch.sum(r1)/len(y)
    return r2

###Forward Pass
Wx = torch.matmul(W,x)
sig_Wx = sig(Wx)
#fwd_prop = torch.linalg.matrix_norm(sig_Wx,2)
fwd_prop = MSE(yhat,sig_Wx)
print(fwd_prop)

###Retain gradients for reference
Wx.retain_grad()
sig_Wx.retain_grad()
fwd_prop.retain_grad()

###Backwards Prop
fwd_prop.backward()

###Gradient Results
print("W Backprop Result:", W.grad)
print("x Backprop Result:", x.grad)