#load libraries
import torch
from torch.autograd import Variable


# build variable
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
#print(tensor)
#print(variable) # from here it can be seen that tensor and variable are same but variable is part of auto gradient
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
#print(t_out)
#print(v_out)
v_out.backward()
#print(variable.grad)


#address vriable in different format
#print(variable) # this is variable format
#print(variable.data) # this is tensor format
#print(variable.data.numpy()) # this is numpy format




'''
Content:
1) build variable
2) address variable in different type
'''