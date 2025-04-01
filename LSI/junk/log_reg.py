import torch
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.flatten(x)
        outputs = torch.sigmoid(self.linear(x))
        return outputs
    
    def freeze_all_but_last(self):
        return