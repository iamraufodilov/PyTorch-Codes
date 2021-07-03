# load libraries
import torch
import torch.nn.functional as F
import matplotlib as plt
from torch.autograd import Variable

torch.manual_seed(1)


# create x and y dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())
# torch only can get variable input so conver x and y data to variable
x, y = Variable(x), Variable(y)


# create class
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
#print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss() # this is for regression mean squared loss

#plt.ion()

for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()