import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGG16Wrapper(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VGG16Wrapper, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.num_classes = num_classes
        
        # Modify the classifier to match the number of classes
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.vgg16(x)
        x = F.softmax(x, dim=1)
        return x