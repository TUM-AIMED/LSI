from models.basic_mlp import MLPMultiLabel
from models.vgg16 import VGG16Wrapper
from models.alexNet import AlexNetWrapper

def get_model(keyword):
    if keyword == "mlp":
        return MLPMultiLabel
    elif keyword == "vgg16":
        return VGG16Wrapper
    elif keyword == "alexnet":
        return AlexNetWrapper
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")