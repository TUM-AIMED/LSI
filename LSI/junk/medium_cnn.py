import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn

class CustomMedCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMedCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(2)
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.features = torch.nn.Sequential(self.conv1, self.relu1, self.max_pool,
                                            self.conv2, self.relu1, self.max_pool,
                                            self.conv3, self.relu1,
                                            self.flatten, 
                                            self.fc1, self.relu1, 
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