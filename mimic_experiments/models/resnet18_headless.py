import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18WrapperHeadless(nn.Module):
    def __init__(self):
        super(ResNet18WrapperHeadless, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x
    
    def freeze_all_but_last(self):
        #named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if 'fc' in n:
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
