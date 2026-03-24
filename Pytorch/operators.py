import torch

#Compatible shapes
a = torch.tensor([[1,1],[2,2]])
b = torch.tensor([[2,2],[3,3]])
print(a+b) # elementwise addition
print(a*b) # elementwise multiplication


#incompatible shapes
a = torch.tensor([[1,1],[2,2]])
c = torch.tensor([[2,2,4],[3,3,5]])
print(a+c) # this will throw an error because the shapes are not compatible for addition


#Matrix multiplication
print(a@b) # this will perform matrix multiplication