# load libraries 
import numpy as np
import torch


# conert numpy to tensor or vice versia
np_data = np.arange(6).reshape(2,3)
#print(np_data)
torch_data = torch.from_numpy(np_data)
#print(torch_data)
tensor2array = torch_data.numpy()
#print(tensor2array)


# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)
#print(tensor)
adding = np.matmul(data, data)
tadding = torch.mm(tensor, tensor)
print(adding)
print(tadding)


'''
Content:

1) conert numpy to tensor or vice versia
2) matrix multiplication
3) 
'''