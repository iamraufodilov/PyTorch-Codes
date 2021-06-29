# load libraries
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# fake data
x = torch.linspace(-5, 5, 200)
#print(len(x))
x = Variable(x)
x_np = x.data.numpy()
#print(x_np)


# following are popular activtion functions
y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
y_softmax = torch.softmax(x, dim=0).data.numpy()


# visualise activation functions
plt.figure(1, figsize=(8,6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim(-0.2, 6)
plt.legend(loc='best')

plt.show()


'''
Content:
1) popular activtion functions
2) visualize activation functions
'''