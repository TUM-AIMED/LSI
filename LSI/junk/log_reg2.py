import torch
class LogisticRegression2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression2, self).__init__()
        hidden_dim = int(input_dim/2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(input_dim, input_dim)
        self.linear2 = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        outputs = torch.sigmoid(x)
        return outputs
    
    def freeze_all_but_last(self):
        return