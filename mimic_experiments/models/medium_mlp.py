import torch
import torch.nn.functional as F
import torch.nn as nn

class MEDIUMMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MEDIUMMLP, self).__init__()
        hidden_dim = 128
        hidden_dim2 = 64
        self.avgpool = torch.nn.AvgPool2d(2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear1 = torch.nn.Linear(int(input_dim/4), hidden_dim)
        self.Sigmoid = torch.nn.Sigmoid()
        self.ReLu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.features = torch.nn.Sequential(self.avgpool, self.flatten, self.linear1, self.Sigmoid, self.linear2, self.Sigmoid, self.linear3)


    def forward(self, x):
        output = self.features(x)
        # output = self.softmax(output) # Put softmax out o fsequential due to la.fit - no idea if this is correct
        return output
 
