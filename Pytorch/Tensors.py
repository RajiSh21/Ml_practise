import torch

my_list = [[1,2,3],[4,5,6]]
tensor = torch.tensor(my_list)
print (tensor)
print(tensor.shape) # shape of the tensor
print(tensor.dtype) # data type of the tensor