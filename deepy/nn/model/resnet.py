import torch
import torch.nn as nn
import torch.nn.functional as F
from deepy.nn import layer


class ResNet(nn.Module):
    """ResNet18, 34, 50, 101, and 152

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ResNet('ResNet18').to(device)
    >>> summary(net , (3, 32, 32))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 32, 32]           1,728
           BatchNorm2d-2           [-1, 64, 32, 32]             128
                Conv2d-3           [-1, 64, 32, 32]          36,864
           BatchNorm2d-4           [-1, 64, 32, 32]             128
                Conv2d-5           [-1, 64, 32, 32]          36,864
           BatchNorm2d-6           [-1, 64, 32, 32]             128
            BasicBlock-7           [-1, 64, 32, 32]               0
                Conv2d-8           [-1, 64, 32, 32]          36,864
           BatchNorm2d-9           [-1, 64, 32, 32]             128
               Conv2d-10           [-1, 64, 32, 32]          36,864
          BatchNorm2d-11           [-1, 64, 32, 32]             128
           BasicBlock-12           [-1, 64, 32, 32]               0
               Conv2d-13          [-1, 128, 16, 16]          73,728
          BatchNorm2d-14          [-1, 128, 16, 16]             256
               Conv2d-15          [-1, 128, 16, 16]         147,456
          BatchNorm2d-16          [-1, 128, 16, 16]             256
               Conv2d-17          [-1, 128, 16, 16]           8,192
          BatchNorm2d-18          [-1, 128, 16, 16]             256
           BasicBlock-19          [-1, 128, 16, 16]               0
               Conv2d-20          [-1, 128, 16, 16]         147,456
          BatchNorm2d-21          [-1, 128, 16, 16]             256
               Conv2d-22          [-1, 128, 16, 16]         147,456
          BatchNorm2d-23          [-1, 128, 16, 16]             256
           BasicBlock-24          [-1, 128, 16, 16]               0
               Conv2d-25            [-1, 256, 8, 8]         294,912
          BatchNorm2d-26            [-1, 256, 8, 8]             512
               Conv2d-27            [-1, 256, 8, 8]         589,824
          BatchNorm2d-28            [-1, 256, 8, 8]             512
               Conv2d-29            [-1, 256, 8, 8]          32,768
          BatchNorm2d-30            [-1, 256, 8, 8]             512
           BasicBlock-31            [-1, 256, 8, 8]               0
               Conv2d-32            [-1, 256, 8, 8]         589,824
          BatchNorm2d-33            [-1, 256, 8, 8]             512
               Conv2d-34            [-1, 256, 8, 8]         589,824
          BatchNorm2d-35            [-1, 256, 8, 8]             512
           BasicBlock-36            [-1, 256, 8, 8]               0
               Conv2d-37            [-1, 512, 4, 4]       1,179,648
          BatchNorm2d-38            [-1, 512, 4, 4]           1,024
               Conv2d-39            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-40            [-1, 512, 4, 4]           1,024
               Conv2d-41            [-1, 512, 4, 4]         131,072
          BatchNorm2d-42            [-1, 512, 4, 4]           1,024
           BasicBlock-43            [-1, 512, 4, 4]               0
               Conv2d-44            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-45            [-1, 512, 4, 4]           1,024
               Conv2d-46            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-47            [-1, 512, 4, 4]           1,024
           BasicBlock-48            [-1, 512, 4, 4]               0
               Linear-49                   [-1, 10]           5,130
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 11.25
    Params size (MB): 42.63
    Estimated Total Size (MB): 53.89
    ----------------------------------------------------------------
    """
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1,
                     down_sampling_layer=nn.Conv2d):
            super(ResNet.BasicBlock, self).__init__()
            if stride != 1:
                self.conv1 = down_sampling_layer(
                    in_planes, planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)
            else:
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False)

            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1:
                self.shortcut = nn.Sequential(
                    down_sampling_layer(
                        in_planes, self.expansion*planes, kernel_size=1,
                        stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1,
                     down_sampling_layer=nn.Conv2d):
            super(ResNet.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes,
                                   kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            if stride != 1:
                self.conv2 = down_sampling_layer(
                    planes, planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)
            else:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1:
                self.shortcut = nn.Sequential(
                    down_sampling_layer(
                        in_planes, self.expansion*planes, kernel_size=1,
                        stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    def __init__(self, resnet_name, num_classes=10,
                 down_sampling_layer=nn.Conv2d):
        super(ResNet, self).__init__()
        if resnet_name == "ResNet18":
            block = ResNet.BasicBlock
            num_blocks = [2, 2, 2, 2]
        elif resnet_name == "ResNet34":
            block = ResNet.BasicBlock
            num_blocks = [3, 4, 6, 3]
        elif resnet_name == "ResNet50":
            block = ResNet.Bottleneck
            num_blocks = [3, 4, 6, 3]
        elif resnet_name == "ResNet101":
            block = ResNet.Bottleneck
            num_blocks = [3, 4, 23, 3]
        elif resnet_name == "ResNet152":
            block = ResNet.Bottleneck
            num_blocks = [3, 8, 36, 3]
        else:
            raise NotImplementedError()

        self.in_planes = 64
        self.down_sampling_layer = down_sampling_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                down_sampling_layer=self.down_sampling_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
