import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Model getter
def get_model(model_name):
    models = {
        "MLP": SmallMLP,
        "SmallCNN": SmallCNN,
        "SmallerCNN": SmallerCNN,
        "TinyModel": TinyModel,
        "MiddleModel": MiddleModel,
        "LargerModel": LargerModel,
        "SmallestCNN": SmallestCNN,
        "SmallestCNN_proxy": SmallestCNN_proxy,
        "ResNet": ResNet8,
        "VGG": VGG7,
        "ResNet18": CustomResNet18,
        "ResNet9": ResNet9,
        "SmallerResNet": SmallerResNet,
        "SmallestestCNN": SmallestestCNN
    }
    return models.get(model_name)



# Utility function for convolutional blocks
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels, track_running_stats=False),
              nn.ReLU(inplace=False)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


# Models
class ElementwiseAddition(nn.Module):
    def forward(self, x, shortcut):
        return x + shortcut


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, n_classes, model_name):
        super(ResNet8, self).__init__()
        if model_name in ["cifar10", "cifar100"]:
            self.n_pool = 1
            self.n_fully = 1
        elif model_name == "Prima":
            self.n_pool = 2
            self.n_fully = 64
        elif model_name == "Prima_smaller":
            self.n_pool = 2
            self.n_fully = 4
        elif model_name in ["Imagenette", "Imagewoof"]:
            self.n_pool = 2
            self.n_fully = 4

        block = BasicBlock
        num_blocks = [2, 2, 2]
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion * self.n_fully, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8 * self.n_pool)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SmallMLP(nn.Module):
    def __init__(self, n_classes, model_name):
        super(SmallMLP, self).__init__()
        if model_name in ["cifar10", "cifar100"]:
            in_features = 32 * 32 * 3
        elif model_name == "Prima":
            in_features = 512 * 512 * 3
        elif model_name == "Prima_smaller":
            in_features = 128 * 128 * 3
        elif model_name in ["Imagenette", "Imagewoof"]:
            in_features = 160 * 160 * 3

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.features(x)


class SmallCNN(nn.Module):
    def __init__(self, n_classes, model_name="cifar10"):
        super(SmallCNN, self).__init__()
        if model_name in ["cifar10", "cifar100"]:
            in_features = 64 * 4 * 4
        elif model_name == "Prima":
            in_features = 64 * 64 * 64
        elif model_name == "Prima_smaller":
            in_features = 64 * 16 * 16
        elif model_name in ["Imagenette", "Imagewoof"]:
            in_features = 25600

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x):
        return self.features(x)


class SmallestCNN(nn.Module):
    def __init__(self, n_classes):
        super(SmallestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 16 * 8 * 8 // 2),
            nn.ReLU(),
            nn.Linear(16 * 8 * 8 // 2, n_classes)
        )

    def forward(self, x):
        return self.features(x)


class SmallestestCNN(nn.Module):
    def __init__(self, n_classes):
        super(SmallestestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 8 * 8 * 8 // 2),
            nn.ReLU(),
            nn.Linear(8 * 8 * 8 // 2, n_classes)
        )

    def forward(self, x):
        return self.features(x)


class SmallestCNN_proxy(nn.Module):
    def __init__(self, n_classes):
        super(SmallestCNN_proxy, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(16 * 8 * 8 // 2, n_classes)
        )

    def forward(self, x):
        return self.features(x)


class SmallerCNN(nn.Module):
    def __init__(self, n_classes):
        super(SmallerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.features(x)


class VGG7(nn.Module):
    def __init__(self, n_classes, model_name):
        super(VGG7, self).__init__()
        if model_name in ["cifar10", "cifar100"]:
            in_features = 512
        elif model_name == "Prima":
            in_features = 131072
        elif model_name == "Prima_smaller":
            in_features = 8192
        elif model_name in ["Imagenette", "Imagewoof"]:
            in_features = 12800

        def make_vgg_block(in_channels, out_channels, num_convs):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),
            make_vgg_block(64, 128, 2),
            make_vgg_block(128, 256, 3),
            make_vgg_block(256, 512, 3),
            make_vgg_block(512, 512, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomResNet18(nn.Module):
    def __init__(self, n_classes):
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        return self.model(x)
    
    def pass_except_last(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    


class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super(ResNet9, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class SmallerResNet(nn.Module):
    def __init__(self, num_classes):
        super(SmallerResNet, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x).contiguous()
        out = self.classifier(out)
        return out


class TinyModel(nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(TinyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x):
        return self.features(x.to(torch.float32))


class LargerModel(nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(LargerModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, n_classes)
        )

    def forward(self, x):
        return self.features(x.to(torch.float32))


class MiddleModel(nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(MiddleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, n_classes)
        )

    def forward(self, x):
        return self.features(x.to(torch.float32))


