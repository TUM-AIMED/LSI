import torch
import torch.nn.functional as F


class MLPMultiLabel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPMultiLabel, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.features = torch.nn.Sequential(self.flatten, self.linear1, self.relu1, self.linear2, self.softmax)


    def forward(self, x):
        output = self.features(x)
        return output

    def ReLU_inplace_to_False(self, features):
        for layer in features._modules.values():
            if isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.ELU):
                layer.inplace = False
            self.ReLU_inplace_to_False(layer)