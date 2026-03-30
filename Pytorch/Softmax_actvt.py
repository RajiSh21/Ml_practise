import torch
import torch.nn as nn

#input tensor
input_tensor = torch.tensor([4.3, 6.1, 2.3])

#softmax along the last dimension
probabilities = nn.Softmax(dim=0)
output_tensor = probabilities(input_tensor)
print(output_tensor)