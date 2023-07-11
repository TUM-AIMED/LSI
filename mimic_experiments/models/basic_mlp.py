import torch
import torch.nn.functional as F


class MLPMultiLabel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPMultiLabel, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.linear1(x))
        output = self.linear2(output)
        output = F.softmax(output, dim=1)
        return output
