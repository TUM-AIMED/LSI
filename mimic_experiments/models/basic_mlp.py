import torch
import torch.nn.functional as F
import torch.nn as nn

class MLPMultiLabel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPMultiLabel, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.features = torch.nn.Sequential(self.flatten, self.linear1, self.relu1, self.linear2)


    def forward(self, x):
        output = self.features(x)
        output = self.softmax(output) # Put softmax out o fsequential due to la.fit - no idea if this is correct
        return output
 
    def freeze_all_but_last(self):
        #named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if 'classifier' in n:
                pass
            else:
                p.requires_grad = False

    def test_freeze(self):
        for p in self.parameters():
            print(p.requires_grad)

    def ReLU_inplace_to_False(self, features):
        for layer in features._modules.values():
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.ELU):
                layer.inplace = False
            self.ReLU_inplace_to_False(layer)

    def replace_relu_with_elu(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ELU(inplace=True))
            else:
                self.replace_relu_with_elu(child)

    def replace_all_relu_with_elu(self):
        self.replace_relu_with_elu(self.features)
        return 
    
    def clip_weights(self, clip_value):
        for param_name, param in self.features.named_parameters():
            if 'weight' in param_name:
                param.data = torch.clamp(param.data, min=-clip_value, max=clip_value)