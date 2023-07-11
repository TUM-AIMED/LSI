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
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x