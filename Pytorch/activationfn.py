import torch 
import torch.nn as nn

input_tensor = torch.tensor([[6]])
sigmoid = nn.Sigmoid()
output = sigmoid(input_tensor)
print(output)

model = nn.Sequential(
    nn.Linear(6, 4) #first linear layer takes 6 features and outputs 4 features
    nn.Linear(4, 1) #second linear layer takes 4 features and outputs 1 feature
    nn.sigmoid() #Sigmoid activation function
)

