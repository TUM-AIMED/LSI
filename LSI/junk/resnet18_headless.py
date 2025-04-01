import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict


class ResNet18WrapperHeadless(nn.Module):
    def __init__(self, pretrained="Imagenet"):
        super(ResNet18WrapperHeadless, self).__init__()
        if pretrained == "Imagenet":
            self.resnet = models.resnet18(pretrained=True)
        elif pretrained =="Places365":
            pretrained_weights = torch.load('/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/models/pre_trained_weights/resnet18_places365.pth', map_location=torch.device('cpu'))
            # Create a new ordered dictionary with keys without the "module." prefix
            original_state_dict = pretrained_weights["state_dict"]
            new_state_dict = OrderedDict()

            for key, value in original_state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            
            self.resnet = models.resnet18()
            self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 365)

            self.resnet.load_state_dict(new_state_dict)
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
