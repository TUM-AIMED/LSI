from models.basic_mlp import MLPMultiLabel
from models.vgg16 import VGG16Wrapper
from models.alexNet import AlexNetWrapper
from models.resnet34 import ResNet34Wrapper
from models.resnet18 import ResNet18Wrapper
from models.simple_cnn import CustomCNN
from models.medium_cnn import CustomMedCNN
from models.medium_to_small_cnn import CustomSmallMedCNN
from models.vgg8 import CustomVGG8
from models.small_mlp import SMALLMLP

def get_model(keyword):
    if keyword == "mlp":
        return MLPMultiLabel
    elif keyword =="cnn":
        return CustomCNN
    elif keyword =="med_cnn":
        return CustomMedCNN
    elif keyword == "med_small_cnn":
        return CustomSmallMedCNN
    elif keyword == "vgg16":
        return VGG16Wrapper
    elif keyword == "alexnet":
        return AlexNetWrapper
    elif keyword == "resnet34":
        return ResNet34Wrapper
    elif keyword == "resnet18":
        return ResNet18Wrapper
    elif keyword == "vgg8":
        return CustomVGG8
    elif keyword == "small_mlp":
        return SMALLMLP
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")