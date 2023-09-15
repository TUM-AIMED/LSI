import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class AlexNetWrapper(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AlexNetWrapper, self).__init__()
        self.num_classes = num_classes
        self.features = models.alexnet(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        
        self.features = nn.Sequential(self.features, self.avgpool, self.flatten, self.classifier)

    def forward(self, x):
        x = self.features(x)
        return x
    
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