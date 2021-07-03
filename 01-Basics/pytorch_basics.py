# load libraries
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms

#---------------------------------------------------------------------------------------
# basic autograd example 1
# create tensor
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# build computational graph
y = w*x+b
y.backward()

#print(x.grad)
#print(w.grad)
#print(b.grad)

#---------------------------------------------------------------------------------------------------------
# basic autograd example 2
# create tensor shape of (10,3) and (10,2)
x = torch.randn(10,3)
y = torch.randn(10,2)
#print(x, y)

# build fully connected layer
linear = nn.Linear(3,2)
#print('w', linear.weight)
#print('b', linear.bias)

# build loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.001)

# forward pass
pred = linear(x)

# compute loss
loss = criterion(pred, y)
#print('loss: ', loss.item())

# backward pass
loss.backward()

# print out gradients
#print('w', linear.weight.grad)
#print('b', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

# print out result after gradient descent
pred = linear(x)
loss = criterion(pred, y)
#print('loss after gradient descent', loss.item())


#----------------------------------------------------------------------------------------
# load data from numpy
x = np.array([[1,2], [3,4]])
y = torch.from_numpy(x)
z = y.numpy()
#print(x, y, z)


#----------------------------------------------------------------------------------------
# input pipeline











'''
1) basic autograd example 1
2) basic autograd example 2
3) load data from numpy
4) input pipeline
'''