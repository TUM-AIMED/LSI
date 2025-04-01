import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(2)
        self.features = torch.nn.Sequential(self.conv1, self.tanh, self.avgpool, 
                                            self.conv2, self.tanh, self.avgpool, 
                                            self.flatten, 
                                            self.fc1, self.tanh, 
                                            self.fc2 )

    def forward(self, x):
        output = self.features(x)
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