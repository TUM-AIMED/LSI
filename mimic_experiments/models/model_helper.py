from models.basic_mlp import MLPMultiLabel
from models.vgg16 import VGG16Wrapper
from models.alexNet import AlexNetWrapper
from models.resnet34 import ResNet34Wrapper
from models.resnet18 import ResNet18Wrapper

def get_model(keyword):
    if keyword == "mlp":
        return MLPMultiLabel
    elif keyword == "vgg16":
        return VGG16Wrapper
    elif keyword == "alexnet":
        return AlexNetWrapper
    elif keyword == "resnet34":
        return ResNet34Wrapper
    elif keyword == "resnet18":
        return ResNet18Wrapper
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")