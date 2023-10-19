import torch
import torch.nn.functional as F
import torch.nn as nn

class SMALLMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SMALLMLP, self).__init__()
        hidden_dim = 100
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.Sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.features = torch.nn.Sequential(self.flatten, self.linear1, self.Sigmoid, self.linear2)


    def forward(self, x):
        output = self.features(x)
        # output = self.softmax(output) # Put softmax out o fsequential due to la.fit - no idea if this is correct
        return output
 
