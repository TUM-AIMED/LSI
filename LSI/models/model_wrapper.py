import torch.nn as nn

class CompressionWrapper(nn.Module):
    def __init__(self, model):
        super(CompressionWrapper, self).__init__()
        self.model = model
        # Extract all layers except the last one (assumes the last layer is the classification layer)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.model(x)

    def compress(self, x):
        # Forward pass through the feature extractor (all layers except the last one)
        return self.feature_extractor(x)
    
class ProxyWrapper(nn.Module):
    def __init__(self, model):
        super(ProxyWrapper, self).__init__()
        self.model = model
        # Extract all layers except the last one (assumes the last layer is the classification layer)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.classification_head = list(model.children())[-1]
        self.features =  self.classification_head

    def forward(self, x):
        return self.features(x)
