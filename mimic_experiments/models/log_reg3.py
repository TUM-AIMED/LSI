import torch
class LogisticRegression3(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression3, self).__init__()
        hidden_dim = 256 # int(input_dim/2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(input_dim, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, output_dim)
        # self.linear3 = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        outputs = self.linear3(x)
        return outputs
    
    def freeze_all_but_last(self):
        return