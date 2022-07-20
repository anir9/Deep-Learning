import numpy as np
import torch

###Inputs
x1 = torch.tensor([-np.pi/2], requires_grad=True)
w1 = torch.tensor([1.0], requires_grad=True)
x2 = torch.tensor([0.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)

###Forward Pass - x1,w1
xw1 = torch.mul(x1,w1)
xw12 = torch.mul(xw1,2.0)
cos_xw1 = torch.cos(xw12)
n_cos_xw1 = torch.mul(cos_xw1, -1.0)
n1_cos_xw1 = torch.add(n_cos_xw1, 1.0)
xw1_done = torch.mul(n1_cos_xw1,0.5)

###Forward Pass - x2,w2
xw2 = torch.mul(x2,w2)
cos_xw2 = torch.cos(xw2)


##Forward Pass - Combine
comb = torch.add(xw1_done, cos_xw2)
comb2 = torch.add(comb, 2.0)
final = torch.div(1,comb2)

##Retain Gradients
xw1.retain_grad()
xw12.retain_grad()
cos_xw1.retain_grad()
n_cos_xw1.retain_grad()
n1_cos_xw1.retain_grad()
xw1_done.retain_grad()

xw2.retain_grad()
cos_xw2.retain_grad()

comb.retain_grad()
comb2.retain_grad()
final.retain_grad()

###Backwards Prop
final.backward()

##Gradients
print("x1 Backprop Result:", x1.grad)
print("w1 Backprop Result:", w1.grad)
print("x2 Backprop Result:", x2.grad)
print("w2 Backprop Result:", w2.grad)