import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.features = torch.nn.Sequential(self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.shortcut, self.relu2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out  + self.shortcut(x)
        out = F.relu(out)
        # out = self.features(x)
        return out

class ResNet8(nn.Module):
    def __init__(self, n_classes, model_name):
        if model_name == "cifar10" or model_name == "cifar100":
            self.n_pool = 1
            self.n_fully = 1
        elif model_name == "Prima":
            self.n_pool = 2
            self.n_fully = 64
        elif model_name == "Prima_smaller":
            self.n_pool = 2
            self.n_fully = 4
        elif model_name == "Imagenette" or model_name == "Imagewoof":
            self.n_pool = 2
            self.n_fully = 4
        block = BasicBlock
        num_blocks = [2, 2, 2]
        super(ResNet8, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion * self.n_fully, n_classes)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = F.avg_pool2d(out, 8*self.n_pool)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.features(x)
        return out

def ResNet8_small():
    return ResNet8(BasicBlock, [2, 2, 2])


import torch.nn as nn

class SmallMLP(nn.Module):
    def __init__(self, n_classes, model_name):
        if model_name == "cifar10" or model_name == "cifar100":
            in_features = 32 * 32 * 3
        elif model_name == "Prima":
            in_features = 512 * 512 * 3
        elif model_name == "Prima_smaller":
            in_features = 128 * 128 * 3
        elif model_name == "Imagenette" or model_name == "Imagewoof":
            in_features = 160 * 160 * 3
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(self.flatten, self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.relu3, self.fc4)

    def forward(self, x):
        # x = x.view(-1, 32 * 32 * 3)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        x = self.features(x)
        return x

import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes, model_name="cifar10"):
        if model_name == "cifar10" or model_name == "cifar100":
            in_features = 64 * 4 * 4
        elif model_name == "Prima":
            in_features = 64 * 64 * 64
        elif model_name == "Prima_smaller":
            in_features = 64 * 16 * 16
        elif model_name == "Imagenette" or model_name == "Imagewoof":
            in_features = 25600
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(in_features, n_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(self.conv1, self.relu1, self.pool1, self.conv2, self.relu2, self.pool2, self.conv3, self.relu3, self.pool3, 
                                      self.flatten, self.fc2)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 64 * 4 * 4)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.features(x)
        return x
    
class SmallestCNN(nn.Module):
    def __init__(self, n_classes):
        super(SmallestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2
            
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(in_features=16*8*8, out_features=int(16*8*8/2)),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(in_features=int(16*8*8/2), out_features=n_classes)  # Fully connected layer
        )
    
    def forward(self, x):
        return self.features(x)
    
    def compress(self, x):
        new_model_layers = list(self.features.children())[:-1]
        self.proxy = nn.Sequential(*new_model_layers)
        x = self.proxy(x)
        return x
    
class SmallestestCNN(nn.Module):
    def __init__(self, n_classes):
        super(SmallestestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1
            
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2
            
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(in_features=8*8*8, out_features=int(8*8*8/2)),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(in_features=int(8*8*8/2), out_features=n_classes)  # Fully connected layer
        )
    
    def forward(self, x):
        return self.features(x)

class SmallestCNN_proxy_old(nn.Module):
    def __init__(self, n_classes):
        super(SmallestCNN_proxy_old, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=16*8*8, out_features=int(16*8*8/2)),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(in_features=int(16*8*8/2), out_features=n_classes)  # Fully connected layer
        )
    
    def forward(self, x):
        return self.features(x)
    

class SmallestCNN_proxy(nn.Module):
    def __init__(self, n_classes):
        super(SmallestCNN_proxy, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=int(16*8*8/2), out_features=n_classes)  # Fully connected layer
        )
    
    def forward(self, x):
        return self.features(x)
    

class SmallerCNN(nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(SmallerCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=n_classes)
        self.features = nn.Sequential(self.conv1, self.relu1, self.pool1, self.conv2, self.relu2, self.pool2, self.conv3, self.relu3, self.pool3, 
                                      self.flatten, self.fc1, self.relu4, self.fc2, self.relu5, self.fc3)

        
        # Dropout layer

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(-1, 128 * 4 * 4)
        
        # Apply fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
class VGG7(nn.Module):
    def __init__(self, n_classes, model_name):
        if model_name == "cifar10" or model_name == "cifar100":
            in_features = 512
        elif model_name == "Prima":
            in_features = 131072
        elif model_name == "Prima_smaller":
            in_features = 8192
        elif model_name == "Imagenette" or model_name == "Imagewoof":
            in_features = 12800
        super(VGG7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class CustomResNet18(nn.Module):
    def __init__(self, n_classes):
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)
    
    def forward(self, x):
        return self.model(x)
    
    
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels, track_running_stats=False),
            #   nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=False)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        in_channels = 3
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out_res1 = self.res1(out)  # Store result of residual block
        out = out_res1 + out       # Explicit addition (no in-place modification)
        out = self.conv3(out)
        out = self.conv4(out)
        out_res2 = self.res2(out)  # Store result of residual block
        out = out_res2 + out       # Explicit addition (no in-place modification)
        out = self.classifier(out)
        return out
    
class SmallerResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        in_channels = 3
        self.conv1 = conv_block(in_channels, 64)

        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(64, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb).contiguous()
        out = self.classifier(out)
        return out
    
class TinyModel(torch.nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, n_classes)
        self.features = torch.nn.Sequential(self.linear1)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x


class LargerModel(torch.nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(LargerModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, int(in_features/2))
        self.linear2 = torch.nn.Linear(int(in_features/2), n_classes)
        self.relu = torch.nn.ReLU()
        self.features = torch.nn.Sequential(self.linear1,
                                            self.relu,
                                            self.linear2)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MiddleModel(torch.nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(MiddleModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, int(in_features/16))
        self.linear2 = torch.nn.Linear(int(in_features/16), n_classes)
        self.relu = torch.nn.ReLU()
        self.features = torch.nn.Sequential(self.linear1,
                                            self.relu,
                                            self.linear2)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def get_model(model_name):
    if model_name == "MLP":
        return SmallMLP
    elif model_name == "SmallCNN":
        return SmallCNN
    elif model_name == "SmallerCNN":
        return SmallerCNN
    elif model_name == "TinyModel":
        return TinyModel
    elif model_name == "MiddleModel":
        return MiddleModel
    elif model_name == "LargerModel":
        return LargerModel
    elif model_name == "SmallestCNN":
        return SmallestCNN
    elif model_name == "SmallestCNN_proxy":
        return SmallestCNN_proxy
    elif model_name == "ResNet":
        return ResNet8
    elif model_name == "VGG":
        return VGG7
    elif model_name == "ResNet18":
        return CustomResNet18
    elif model_name == "ResNet9":
        return ResNet9
    elif model_name == "SmallerResNet":
        return SmallerResNet
    elif model_name == "SmallestestCNN":
        return SmallestestCNN
