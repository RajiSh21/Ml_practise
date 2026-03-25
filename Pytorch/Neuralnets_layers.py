import torch.nn as nn
import torch

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]]) #input tensor with 3 features

#linear layer
linear_layer = nn.Linear(in_features =3, out_features=2)

#pass input through linear layer
output = linear_layer(input_tensor)
print(linear_layer.weight) #weights of the linear layer
print(linear_layer.bias) #bias of the linear layer
print(output)
