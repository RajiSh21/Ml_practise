import torch as nn
 

 model =nn.Sequential(
  nn.Linear(n_features, 8) #n_features is the number of features in the input data
  nn.Linear(8, 4 )
  nn.Linear(4, n_classes) #n_classes is the number of classes in the output data
 )


#creating network with threee linear layer
model = nn.Sequential(
    nn.Linear(10, 18) #takes 10 feature and outputs 18 feature
    nn.Linear(18, 20), #takes 18 feature and outputs 20 feature
    nn.Linear(20, 5) #takes 20 feature and outputs 5 feature
)
 
 #calculate using .numel() method
total =0
for parameter in model.parameters():
 total += parameter.numel()
 print(total)