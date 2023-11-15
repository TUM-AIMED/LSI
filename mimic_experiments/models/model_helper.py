from models.basic_mlp import MLPMultiLabel
from models.vgg16 import VGG16Wrapper
from models.alexNet import AlexNetWrapper
from models.resnet34 import ResNet34Wrapper
from models.resnet18 import ResNet18Wrapper
from models.resnet18_extra_head import ResNet18Wrapper as ResNet18_extra
from models.resnet18_headless import ResNet18WrapperHeadless
from models.simple_cnn import CustomCNN
from models.medium_cnn import CustomMedCNN
from models.medium_to_small_cnn import CustomSmallMedCNN
from models.vgg8 import CustomVGG8
from models.small_mlp import SMALLMLP
from models.medium_mlp import MEDIUMMLP
from models.log_reg import LogisticRegression
from models.log_reg2 import LogisticRegression2
from models.log_reg3 import LogisticRegression3
from models.log_reg4 import LogisticRegression4

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
    elif keyword == "resnet18e":
        return ResNet18_extra
    elif keyword == "resnet18_headless":
        return ResNet18WrapperHeadless
    elif keyword == "vgg8":
        return CustomVGG8
    elif keyword == "small_mlp":
        return SMALLMLP
    elif keyword == "med_mlp":
        return MEDIUMMLP
    elif keyword == "logreg":
        return LogisticRegression
    elif keyword == "logreg2":
        return LogisticRegression2
    elif keyword == "logreg3":
        return LogisticRegression3
    elif keyword == "logreg4":
        return LogisticRegression4
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")