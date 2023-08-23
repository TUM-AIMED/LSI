import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18Wrapper(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNet18Wrapper, self).__init__()
        self.num_classes = num_classes
        self.features = models.resnet18(pretrained=True)
        in_features = self.features.fc.in_features
        self.features.fc = nn.Linear(in_features, num_classes)

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