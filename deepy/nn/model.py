import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from deepy import layer


class MyNet(nn.Module):
    """A simple CNN

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = MyNet().to(device)
    >>> summary(net , (3, 32, 32))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 16, 16]           1,792
           BatchNorm2d-2           [-1, 64, 16, 16]             128
                  ReLU-3           [-1, 64, 16, 16]               0
                Conv2d-4            [-1, 128, 8, 8]          73,856
           BatchNorm2d-5            [-1, 128, 8, 8]             256
                  ReLU-6            [-1, 128, 8, 8]               0
                Conv2d-7            [-1, 256, 4, 4]         295,168
           BatchNorm2d-8            [-1, 256, 4, 4]             512
                  ReLU-9            [-1, 256, 4, 4]               0
               Conv2d-10            [-1, 512, 2, 2]       1,180,160
          BatchNorm2d-11            [-1, 512, 2, 2]           1,024
                 ReLU-12            [-1, 512, 2, 2]               0
               Conv2d-13             [-1, 10, 1, 1]          20,490
    ================================================================
    Total params: 1,573,386
    Trainable params: 1,573,386
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 0.70
    Params size (MB): 6.00
    Estimated Total Size (MB): 6.72
    ----------------------------------------------------------------
    """
    def __init__(self, down_sampling_layer=nn.Conv2d):
        super(MyNet, self).__init__()
        self.net = nn.Sequential(
            down_sampling_layer(in_channels=3, out_channels=64,
                            kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            down_sampling_layer(in_channels=64, out_channels=128,
                            kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            down_sampling_layer(in_channels=128, out_channels=256,
                            kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            down_sampling_layer(in_channels=256, out_channels=512,
                            kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=10,
                      kernel_size=2, stride=1, padding=0)
            )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


class VGG(nn.Module):
    """VGG8, 11, 13, 16, and 19
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = VGG('VGG8').to(device)
    >>> summary(net , (3, 32, 32))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 32, 32]           1,792
           BatchNorm2d-2           [-1, 64, 32, 32]             128
                  ReLU-3           [-1, 64, 32, 32]               0
                Conv2d-4           [-1, 64, 16, 16]          36,928
                Conv2d-5          [-1, 128, 16, 16]          73,856
           BatchNorm2d-6          [-1, 128, 16, 16]             256
                  ReLU-7          [-1, 128, 16, 16]               0
                Conv2d-8            [-1, 128, 8, 8]         147,584
                Conv2d-9            [-1, 256, 8, 8]         295,168
          BatchNorm2d-10            [-1, 256, 8, 8]             512
                 ReLU-11            [-1, 256, 8, 8]               0
               Conv2d-12            [-1, 256, 4, 4]         590,080
               Conv2d-13            [-1, 512, 4, 4]       1,180,160
          BatchNorm2d-14            [-1, 512, 4, 4]           1,024
                 ReLU-15            [-1, 512, 4, 4]               0
               Conv2d-16            [-1, 512, 2, 2]       2,359,808
               Conv2d-17            [-1, 512, 2, 2]       2,359,808
          BatchNorm2d-18            [-1, 512, 2, 2]           1,024
                 ReLU-19            [-1, 512, 2, 2]               0
               Conv2d-20            [-1, 512, 1, 1]       2,359,808
            AvgPool2d-21            [-1, 512, 1, 1]               0
               Linear-22                   [-1, 10]           5,130
    ================================================================
    Total params: 9,413,066
    Trainable params: 9,413,066
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 3.10
    Params size (MB): 35.91
    Estimated Total Size (MB): 39.02
    ----------------------------------------------------------------
    """
    def __init__(self, vgg_name, down_sampling_layer=nn.Conv2d):
        super(VGG, self).__init__()
        self.CFG = {
            'VGG8': [64, 'D', 128, 'D', 256, 'D', 512, 'D', 512, 'D'],
            'VGG11': [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512, 'D'],
            'VGG13': [64, 64, 'D', 128, 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512, 'D'],
            'VGG16': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 'D', 512, 512, 512, 'D', 512, 512, 512, 'D'],
            'VGG19': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 256, 'D', 512, 512, 512, 512, 'D', 512, 512, 512, 512, 'M'],
        }
        self.down_sampling_layer = down_sampling_layer
        self.features = self._make_layers(self.CFG[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'D':
                layers += [self.down_sampling_layer(
                    in_channels, in_channels,
                    kernel_size=3, stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


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


class DCGANGenerator(nn.Module):
    """Generator of DCGAN
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = DCGANGenerator(10).to(device)
    >>> summary(net , (10, ))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
       ConvTranspose2d-1            [-1, 512, 2, 2]          81,920
           BatchNorm2d-2            [-1, 512, 2, 2]           1,024
             LeakyReLU-3            [-1, 512, 2, 2]               0
       ConvTranspose2d-4            [-1, 512, 4, 4]       4,194,304
           BatchNorm2d-5            [-1, 512, 4, 4]           1,024
             LeakyReLU-6            [-1, 512, 4, 4]               0
       ConvTranspose2d-7            [-1, 512, 8, 8]       4,194,304
           BatchNorm2d-8            [-1, 512, 8, 8]           1,024
             LeakyReLU-9            [-1, 512, 8, 8]               0
      ConvTranspose2d-10          [-1, 256, 16, 16]       2,097,152
          BatchNorm2d-11          [-1, 256, 16, 16]             512
            LeakyReLU-12          [-1, 256, 16, 16]               0
      ConvTranspose2d-13          [-1, 128, 32, 32]         524,288
          BatchNorm2d-14          [-1, 128, 32, 32]             256
            LeakyReLU-15          [-1, 128, 32, 32]               0
      ConvTranspose2d-16           [-1, 64, 64, 64]         131,072
          BatchNorm2d-17           [-1, 64, 64, 64]             128
            LeakyReLU-18           [-1, 64, 64, 64]               0
               Conv2d-19            [-1, 3, 64, 64]           1,728
                 Tanh-20            [-1, 3, 64, 64]               0
    ================================================================
    Total params: 11,228,736
    Trainable params: 11,228,736
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 11.67
    Params size (MB): 42.83
    Estimated Total Size (MB): 54.51
    ----------------------------------------------------------------

    """

    def __init__(self, latent_dim, trans_conv_layer=nn.ConvTranspose2d):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            trans_conv_layer(in_channels=latent_dim, out_channels=512,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            trans_conv_layer(in_channels=512, out_channels=512,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            trans_conv_layer(in_channels=512, out_channels=512,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            trans_conv_layer(in_channels=512, out_channels=256,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            trans_conv_layer(in_channels=256, out_channels=128,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            trans_conv_layer(in_channels=128, out_channels=64,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=3,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), self.latent_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Discriminator of DCGAN
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = DCGANDiscriminator().to(device)
    >>> summary(net , (3, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 32, 32]             384
           BatchNorm2d-2            [-1, 8, 32, 32]              16
             LeakyReLU-3            [-1, 8, 32, 32]               0
                Conv2d-4           [-1, 32, 16, 16]           4,096
           BatchNorm2d-5           [-1, 32, 16, 16]              64
             LeakyReLU-6           [-1, 32, 16, 16]               0
                Conv2d-7             [-1, 64, 8, 8]          32,768
           BatchNorm2d-8             [-1, 64, 8, 8]             128
             LeakyReLU-9             [-1, 64, 8, 8]               0
               Conv2d-10             [-1, 64, 4, 4]          65,536
          BatchNorm2d-11             [-1, 64, 4, 4]             128
            LeakyReLU-12             [-1, 64, 4, 4]               0
               Conv2d-13              [-1, 1, 1, 1]           1,024
    ================================================================
    Total params: 104,144
    Trainable params: 104,144
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.49
    Params size (MB): 0.40
    Estimated Total Size (MB): 0.94
    ----------------------------------------------------------------
    """
    def __init__(self, down_conv_layer=nn.Conv2d, normalization="BN"):
        super(DCGANDiscriminator, self).__init__()
        if normalization == "BN":
            norm = nn.BatchNorm2d
        else:
            raise NotImplementedError()

        self.net = nn.Sequential(
            down_conv_layer(in_channels=3, out_channels=8,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=8, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            down_conv_layer(in_channels=8, out_channels=32,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=32, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            down_conv_layer(in_channels=32, out_channels=64,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=64, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            down_conv_layer(in_channels=64, out_channels=64,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=64, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=1,
                      padding=0, kernel_size=(4, 4),
                      stride=(1, 1), bias=False),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))


class DCGANGeneratorCIFER10(nn.Module):
    """Generator of DCGAN for CIFER10
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = DCGANGeneratorCIFER10(10).to(device)
    >>> summary(net , (10, ))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                 [-1, 8192]          90,112
       ConvTranspose2d-2            [-1, 256, 8, 8]       2,097,408
           BatchNorm2d-3            [-1, 256, 8, 8]             512
                  ReLU-4            [-1, 256, 8, 8]               0
          UpConvBNReLU-5            [-1, 256, 8, 8]               0
       ConvTranspose2d-6          [-1, 128, 16, 16]         524,416
           BatchNorm2d-7          [-1, 128, 16, 16]             256
                  ReLU-8          [-1, 128, 16, 16]               0
          UpConvBNReLU-9          [-1, 128, 16, 16]               0
      ConvTranspose2d-10           [-1, 64, 32, 32]         131,136
          BatchNorm2d-11           [-1, 64, 32, 32]             128
                 ReLU-12           [-1, 64, 32, 32]               0
         UpConvBNReLU-13           [-1, 64, 32, 32]               0
               Conv2d-14            [-1, 3, 32, 32]           1,731
                 Tanh-15            [-1, 3, 32, 32]               0
    ================================================================
    Total params: 2,845,699
    Trainable params: 2,845,699
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 3.61
    Params size (MB): 10.86
    Estimated Total Size (MB): 14.46
    ----------------------------------------------------------------
    """
    class UpConvBNReLU(nn.Module):
        def __init__(self, in_channels, out_channels,
                     kernel_size, stride, padding=0,
                     num_classes=0, up_sampling_layer=nn.ConvTranspose2d):
            super(DCGANGeneratorCIFER10.UpConvBNReLU, self).__init__()
            self.up = up_sampling_layer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding)

            if num_classes == 0:
                self.bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.bn = layer.ConditionalBatchNorm2d(
                            num_features=out_channels,
                            num_classes=num_classes)

            self.relu = nn.ReLU(inplace=True)

        def forward(self, inputs, label_onehots=None):
            x = self.up(inputs)
            if label_onehots is not None:
                x = self.bn(x, label_onehots)
            else:
                x = self.bn(x)
            return self.relu(x)

    def __init__(self, latent_dim, num_classes=0,
                 output_shape=(3, 32, 32),
                 up_sampling_layer=nn.ConvTranspose2d):
        super(DCGANGeneratorCIFER10, self).__init__()
        self.latent_dim = latent_dim
        self.mg = output_shape[1] // 8
        self.dense = nn.Linear(latent_dim, self.mg*self.mg*512)
        self.conv1 = DCGANGeneratorCIFER10.UpConvBNReLU(
                       in_channels=512, out_channels=256,
                       kernel_size=(4, 4), stride=(2, 2),
                       padding=1, num_classes=num_classes,
                       up_sampling_layer=up_sampling_layer)
        self.conv2 = DCGANGeneratorCIFER10.UpConvBNReLU(
                       in_channels=256, out_channels=128,
                       kernel_size=(4, 4), stride=(2, 2),
                       padding=1, num_classes=num_classes,
                       up_sampling_layer=up_sampling_layer)
        self.conv3 = DCGANGeneratorCIFER10.UpConvBNReLU(
                       in_channels=128, out_channels=64,
                       kernel_size=(4, 4), stride=(2, 2),
                       padding=1, num_classes=num_classes,
                       up_sampling_layer=up_sampling_layer)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, inputs, label_onehots=None):
        x = inputs.view(inputs.size(0), self.latent_dim)
        x = self.dense(x).view(inputs.size(0), 512, self.mg, self.mg)
        x = self.conv1(x, label_onehots)
        x = self.conv2(x, label_onehots)
        x = self.conv3(x, label_onehots)
        return self.out(x)


class DCGANDiscriminatorCIFER10(nn.Module):
    """Discriminator of DCGAN for CIFER10
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = DCGANDiscriminatorCIFER10().to(device)
    >>> summary(net , (3, 32, 32))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
          SpectralNorm-1           [-1, 64, 32, 32]               0
             LeakyReLU-2           [-1, 64, 32, 32]               0
          ConvNormReLU-3           [-1, 64, 32, 32]               0
          SpectralNorm-4           [-1, 64, 16, 16]               0
             LeakyReLU-5           [-1, 64, 16, 16]               0
          ConvNormReLU-6           [-1, 64, 16, 16]               0
          SpectralNorm-7          [-1, 128, 16, 16]               0
             LeakyReLU-8          [-1, 128, 16, 16]               0
          ConvNormReLU-9          [-1, 128, 16, 16]               0
         SpectralNorm-10            [-1, 128, 8, 8]               0
            LeakyReLU-11            [-1, 128, 8, 8]               0
         ConvNormReLU-12            [-1, 128, 8, 8]               0
         SpectralNorm-13            [-1, 256, 8, 8]               0
            LeakyReLU-14            [-1, 256, 8, 8]               0
         ConvNormReLU-15            [-1, 256, 8, 8]               0
         SpectralNorm-16            [-1, 256, 4, 4]               0
            LeakyReLU-17            [-1, 256, 4, 4]               0
         ConvNormReLU-18            [-1, 256, 4, 4]               0
         SpectralNorm-19            [-1, 512, 4, 4]               0
            LeakyReLU-20            [-1, 512, 4, 4]               0
         ConvNormReLU-21            [-1, 512, 4, 4]               0
               Linear-22                    [-1, 1]           8,193
    ================================================================
    Total params: 8,193
    Trainable params: 8,193
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 3.47
    Params size (MB): 0.03
    Estimated Total Size (MB): 3.51
    ----------------------------------------------------------------
    """
    class ConvNormReLU(nn.Module):
        def __init__(self, in_channels, out_channels,
                     kernel_size, stride, padding=0,
                     normalization="BN",
                     lrelu_slope=0.1, down_sampling_layer=nn.Conv2d):
            super(DCGANDiscriminatorCIFER10.ConvNormReLU, self).__init__()

            self.conv = down_sampling_layer(
                          in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding)
            if normalization == "SN":
                self.norm = layer.SpectralNorm(self.conv)
            elif normalization == "BN":
                self.norm = nn.Sequential(
                    self.conv,
                    nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError()

            self.lrelu = nn.LeakyReLU(lrelu_slope, inplace=True)

        def forward(self, inputs):
            return self.lrelu(self.norm(inputs))

    def __init__(self, num_classes=0,
                 input_shape=(3, 32, 32),
                 normalization="SN",
                 down_sampling_layer=nn.Conv2d):
        super(DCGANDiscriminatorCIFER10, self).__init__()
        self.mg = input_shape[1] // 8

        self.conv1 = self.discriminator_block(
            in_channels=3, out_channels=64,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.conv2 = self.discriminator_block(
            in_channels=64, out_channels=128,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.conv3 = self.discriminator_block(
            in_channels=128, out_channels=256,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.conv4 = DCGANDiscriminatorCIFER10.ConvNormReLU(
            in_channels=256, out_channels=512,
            kernel_size=(3, 3), stride=(1, 1), padding=1,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.dense = nn.Linear(self.mg * self.mg * 512, 1)

    def discriminator_block(self, in_channels, out_channels,
                            normalization, down_sampling_layer):
        return nn.Sequential(
            DCGANDiscriminatorCIFER10.ConvNormReLU(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(3, 3), stride=(1, 1), padding=1,
                normalization=normalization,
                down_sampling_layer=down_sampling_layer),
            DCGANDiscriminatorCIFER10.ConvNormReLU(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=(4, 4), stride=(2, 2), padding=1,
                normalization=normalization,
                down_sampling_layer=down_sampling_layer))

    def forward(self, inputs, label_onehots=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        base_feature = x.view(inputs.size(0), -1)
        x = self.dense(base_feature)
        return x.view(inputs.size(0))


class SNGANGenerator(nn.Module):
    pass


class SNGANDiscriminator(nn.Module):
    pass


class ConvVAEEncoder(nn.Module):
    """Encoder of Convolutional VAE

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ConvVAEEncoder(10, device).to(device)
    >>> summary(net , (3, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]             448
                Conv2d-2           [-1, 32, 16, 16]           4,640
                Conv2d-3             [-1, 64, 8, 8]          18,496
                Conv2d-4             [-1, 64, 4, 4]          36,928
                Conv2d-5             [-1, 64, 2, 2]          36,928
                Linear-6                   [-1, 10]           2,570
                Linear-7                   [-1, 10]           2,570
    ================================================================
    Total params: 102,580
    Trainable params: 102,580
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.23
    Params size (MB): 0.39
    Estimated Total Size (MB): 0.67
    ----------------------------------------------------------------
    """
    def __init__(self, latent_dim, device, down_conv_layer=nn.Conv2d):
        super(ConvVAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.conv_1 = down_conv_layer(in_channels=3,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_2 = down_conv_layer(in_channels=16,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_3 = down_conv_layer(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_4 = down_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_5 = down_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.z_mean = nn.Linear(64*2*2, self.latent_dim)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = nn.Linear(64*2*2, self.latent_dim)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, features):
        x = self.conv_1(features)
        x = F.leaky_relu(x)
        #print('conv1 out:', x.size())

        x = self.conv_2(x)
        x = F.leaky_relu(x)
        #print('conv2 out:', x.size())

        x = self.conv_3(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        x = self.conv_4(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        x = self.conv_5(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        z_mean = self.z_mean(x.view(-1, 64*2*2))
        z_log_var = self.z_log_var(x.view(-1, 64*2*2))
        encoded = self.reparameterize(z_mean, z_log_var)

        return z_mean, z_log_var, encoded


class ConvVAEDecoder(nn.Module):
    """Decoder of Convolutional VAE

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ConvVAEDecoder(10).to(device)
    >>> summary(net , (10, ))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                  [-1, 256]           2,816
       ConvTranspose2d-2             [-1, 64, 4, 4]          65,600
       ConvTranspose2d-3             [-1, 64, 8, 8]          65,600
       ConvTranspose2d-4           [-1, 32, 16, 16]          32,800
       ConvTranspose2d-5           [-1, 16, 32, 32]           8,208
       ConvTranspose2d-6            [-1, 3, 64, 64]             771
    ================================================================
    Total params: 175,795
    Trainable params: 175,795
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.32
    Params size (MB): 0.67
    Estimated Total Size (MB): 0.99
    ----------------------------------------------------------------
    """
    def __init__(self, latent_dim, up_conv_layer=nn.ConvTranspose2d):
        super(ConvVAEDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.linear_1 = nn.Linear(self.latent_dim, 64*2*2)

        self.deconv_1 = up_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_2 = up_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_3 = up_conv_layer(in_channels=64,
                                      out_channels=32,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_4 = up_conv_layer(in_channels=32,
                                      out_channels=16,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_5 = up_conv_layer(in_channels=16,
                                      out_channels=3,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

    def forward(self, encoded):
        x = self.linear_1(encoded)
        x = x.view(-1, 64, 2, 2)

        x = self.deconv_1(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.deconv_2(x)
        x = F.leaky_relu(x)
        #print('deconv2 out:', x.size())

        x = self.deconv_3(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.deconv_4(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.deconv_5(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        decoded = torch.sigmoid(x)
        return decoded


class UNet(nn.Module):
    """ U-Net

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = UNet(3, 10).to(device)
    >>> summary(net , (3, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 64, 64]           1,792
           BatchNorm2d-2           [-1, 64, 64, 64]             128
                  ReLU-3           [-1, 64, 64, 64]               0
                Conv2d-4           [-1, 64, 64, 64]          36,928
           BatchNorm2d-5           [-1, 64, 64, 64]             128
                  ReLU-6           [-1, 64, 64, 64]               0
           double_conv-7           [-1, 64, 64, 64]               0
                inconv-8           [-1, 64, 64, 64]               0
                Conv2d-9           [-1, 64, 32, 32]          36,864
               Conv2d-10          [-1, 128, 32, 32]          73,856
          BatchNorm2d-11          [-1, 128, 32, 32]             256
                 ReLU-12          [-1, 128, 32, 32]               0
               Conv2d-13          [-1, 128, 32, 32]         147,584
          BatchNorm2d-14          [-1, 128, 32, 32]             256
                 ReLU-15          [-1, 128, 32, 32]               0
          double_conv-16          [-1, 128, 32, 32]               0
                 down-17          [-1, 128, 32, 32]               0
               Conv2d-18          [-1, 128, 16, 16]         147,456
               Conv2d-19          [-1, 256, 16, 16]         295,168
          BatchNorm2d-20          [-1, 256, 16, 16]             512
                 ReLU-21          [-1, 256, 16, 16]               0
               Conv2d-22          [-1, 256, 16, 16]         590,080
          BatchNorm2d-23          [-1, 256, 16, 16]             512
                 ReLU-24          [-1, 256, 16, 16]               0
          double_conv-25          [-1, 256, 16, 16]               0
                 down-26          [-1, 256, 16, 16]               0
               Conv2d-27            [-1, 256, 8, 8]         589,824
               Conv2d-28            [-1, 512, 8, 8]       1,180,160
          BatchNorm2d-29            [-1, 512, 8, 8]           1,024
                 ReLU-30            [-1, 512, 8, 8]               0
               Conv2d-31            [-1, 512, 8, 8]       2,359,808
          BatchNorm2d-32            [-1, 512, 8, 8]           1,024
                 ReLU-33            [-1, 512, 8, 8]               0
          double_conv-34            [-1, 512, 8, 8]               0
                 down-35            [-1, 512, 8, 8]               0
               Conv2d-36            [-1, 512, 4, 4]       2,359,296
               Conv2d-37            [-1, 512, 4, 4]       2,359,808
          BatchNorm2d-38            [-1, 512, 4, 4]           1,024
                 ReLU-39            [-1, 512, 4, 4]               0
               Conv2d-40            [-1, 512, 4, 4]       2,359,808
          BatchNorm2d-41            [-1, 512, 4, 4]           1,024
                 ReLU-42            [-1, 512, 4, 4]               0
          double_conv-43            [-1, 512, 4, 4]               0
                 down-44            [-1, 512, 4, 4]               0
      ConvTranspose2d-45            [-1, 512, 8, 8]       4,194,304
               Conv2d-46            [-1, 256, 8, 8]       2,359,552
          BatchNorm2d-47            [-1, 256, 8, 8]             512
                 ReLU-48            [-1, 256, 8, 8]               0
               Conv2d-49            [-1, 256, 8, 8]         590,080
          BatchNorm2d-50            [-1, 256, 8, 8]             512
                 ReLU-51            [-1, 256, 8, 8]               0
          double_conv-52            [-1, 256, 8, 8]               0
                   up-53            [-1, 256, 8, 8]               0
      ConvTranspose2d-54          [-1, 256, 16, 16]       1,048,576
               Conv2d-55          [-1, 128, 16, 16]         589,952
          BatchNorm2d-56          [-1, 128, 16, 16]             256
                 ReLU-57          [-1, 128, 16, 16]               0
               Conv2d-58          [-1, 128, 16, 16]         147,584
          BatchNorm2d-59          [-1, 128, 16, 16]             256
                 ReLU-60          [-1, 128, 16, 16]               0
          double_conv-61          [-1, 128, 16, 16]               0
                   up-62          [-1, 128, 16, 16]               0
      ConvTranspose2d-63          [-1, 128, 32, 32]         262,144
               Conv2d-64           [-1, 64, 32, 32]         147,520
          BatchNorm2d-65           [-1, 64, 32, 32]             128
                 ReLU-66           [-1, 64, 32, 32]               0
               Conv2d-67           [-1, 64, 32, 32]          36,928
          BatchNorm2d-68           [-1, 64, 32, 32]             128
                 ReLU-69           [-1, 64, 32, 32]               0
          double_conv-70           [-1, 64, 32, 32]               0
                   up-71           [-1, 64, 32, 32]               0
      ConvTranspose2d-72           [-1, 64, 64, 64]          65,536
               Conv2d-73           [-1, 64, 64, 64]          73,792
          BatchNorm2d-74           [-1, 64, 64, 64]             128
                 ReLU-75           [-1, 64, 64, 64]               0
               Conv2d-76           [-1, 64, 64, 64]          36,928
          BatchNorm2d-77           [-1, 64, 64, 64]             128
                 ReLU-78           [-1, 64, 64, 64]               0
          double_conv-79           [-1, 64, 64, 64]               0
                   up-80           [-1, 64, 64, 64]               0
               Conv2d-81           [-1, 10, 64, 64]             650
              outconv-82           [-1, 10, 64, 64]               0
    ================================================================
    Total params: 22,099,914
    Trainable params: 22,099,914
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 58.81
    Params size (MB): 84.30
    Estimated Total Size (MB): 143.16
    ----------------------------------------------------------------
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_ch, out_ch):
            super(UNet.double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet.inconv, self).__init__()
            self.conv = UNet.double_conv(in_ch, out_ch)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d):
            super(UNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
                UNet.double_conv(in_ch, out_ch)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, up_conv_layer=nn.ConvTranspose2d):
            super(UNet.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                        kernel_size=(4, 4), stride=(2, 2),
                                        padding=1, bias=False)
            self.conv = UNet.double_conv(mid_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet.outconv, self).__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d):
        super(UNet, self).__init__()
        self.inc = UNet.inconv(n_channels, 64)
        self.down1 = UNet.down(64, 128)
        self.down2 = UNet.down(128, 256)
        self.down3 = UNet.down(256, 512)
        self.down4 = UNet.down(512, 512)
        self.up1 = UNet.up(512, 1024, 256)
        self.up2 = UNet.up(256, 512, 128)
        self.up3 = UNet.up(128, 256, 64)
        self.up4 = UNet.up(64, 128, 64)
        self.outc = UNet.outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

class ITMNet(UNet):
    """iTM-Net

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ITMNet(3, 10).to(device)
    >>> summary(net , (3, 128, 128))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 128, 128]           1,792
           BatchNorm2d-2         [-1, 64, 128, 128]             128
                  ReLU-3         [-1, 64, 128, 128]               0
                Conv2d-4         [-1, 64, 128, 128]          36,928
           BatchNorm2d-5         [-1, 64, 128, 128]             128
                  ReLU-6         [-1, 64, 128, 128]               0
           double_conv-7         [-1, 64, 128, 128]               0
                inconv-8         [-1, 64, 128, 128]               0
                Conv2d-9           [-1, 64, 64, 64]          36,864
               Conv2d-10          [-1, 128, 64, 64]          73,856
          BatchNorm2d-11          [-1, 128, 64, 64]             256
                 ReLU-12          [-1, 128, 64, 64]               0
               Conv2d-13          [-1, 128, 64, 64]         147,584
          BatchNorm2d-14          [-1, 128, 64, 64]             256
                 ReLU-15          [-1, 128, 64, 64]               0
          double_conv-16          [-1, 128, 64, 64]               0
                 down-17          [-1, 128, 64, 64]               0
               Conv2d-18          [-1, 128, 32, 32]         147,456
               Conv2d-19          [-1, 256, 32, 32]         295,168
          BatchNorm2d-20          [-1, 256, 32, 32]             512
                 ReLU-21          [-1, 256, 32, 32]               0
               Conv2d-22          [-1, 256, 32, 32]         590,080
          BatchNorm2d-23          [-1, 256, 32, 32]             512
                 ReLU-24          [-1, 256, 32, 32]               0
          double_conv-25          [-1, 256, 32, 32]               0
                 down-26          [-1, 256, 32, 32]               0
               Conv2d-27          [-1, 256, 16, 16]         589,824
               Conv2d-28          [-1, 512, 16, 16]       1,180,160
          BatchNorm2d-29          [-1, 512, 16, 16]           1,024
                 ReLU-30          [-1, 512, 16, 16]               0
               Conv2d-31          [-1, 512, 16, 16]       2,359,808
          BatchNorm2d-32          [-1, 512, 16, 16]           1,024
                 ReLU-33          [-1, 512, 16, 16]               0
          double_conv-34          [-1, 512, 16, 16]               0
                 down-35          [-1, 512, 16, 16]               0
               Conv2d-36            [-1, 512, 8, 8]       2,359,296
               Conv2d-37            [-1, 512, 8, 8]       2,359,808
          BatchNorm2d-38            [-1, 512, 8, 8]           1,024
                 ReLU-39            [-1, 512, 8, 8]               0
               Conv2d-40            [-1, 512, 8, 8]       2,359,808
          BatchNorm2d-41            [-1, 512, 8, 8]           1,024
                 ReLU-42            [-1, 512, 8, 8]               0
          double_conv-43            [-1, 512, 8, 8]               0
                 down-44            [-1, 512, 8, 8]               0
               Conv2d-45          [-1, 3, 128, 128]              81
          BatchNorm2d-46          [-1, 3, 128, 128]               6
                 ReLU-47          [-1, 3, 128, 128]               0
         ConvNormReLU-48          [-1, 3, 128, 128]               0
               Conv2d-49           [-1, 64, 64, 64]           1,728
                 down-50           [-1, 64, 64, 64]               0
               Conv2d-51           [-1, 64, 64, 64]          36,864
          BatchNorm2d-52           [-1, 64, 64, 64]             128
                 ReLU-53           [-1, 64, 64, 64]               0
         ConvNormReLU-54           [-1, 64, 64, 64]               0
               Conv2d-55           [-1, 64, 32, 32]          36,864
                 down-56           [-1, 64, 32, 32]               0
               Conv2d-57           [-1, 64, 32, 32]          36,864
          BatchNorm2d-58           [-1, 64, 32, 32]             128
                 ReLU-59           [-1, 64, 32, 32]               0
         ConvNormReLU-60           [-1, 64, 32, 32]               0
               Conv2d-61           [-1, 64, 16, 16]          36,864
                 down-62           [-1, 64, 16, 16]               0
               Conv2d-63           [-1, 64, 16, 16]          36,864
          BatchNorm2d-64           [-1, 64, 16, 16]             128
                 ReLU-65           [-1, 64, 16, 16]               0
         ConvNormReLU-66           [-1, 64, 16, 16]               0
               Conv2d-67             [-1, 64, 8, 8]          36,864
                 down-68             [-1, 64, 8, 8]               0
               Conv2d-69             [-1, 64, 8, 8]          36,864
          BatchNorm2d-70             [-1, 64, 8, 8]             128
                 ReLU-71             [-1, 64, 8, 8]               0
         ConvNormReLU-72             [-1, 64, 8, 8]               0
               Conv2d-73             [-1, 64, 4, 4]          36,864
                 down-74             [-1, 64, 4, 4]               0
               Conv2d-75             [-1, 64, 1, 1]          65,536
          BatchNorm2d-76             [-1, 64, 1, 1]             128
                 ReLU-77             [-1, 64, 1, 1]               0
         ConvNormReLU-78             [-1, 64, 1, 1]               0
               Conv2d-79            [-1, 512, 8, 8]       2,654,720
          BatchNorm2d-80            [-1, 512, 8, 8]           1,024
                 ReLU-81            [-1, 512, 8, 8]               0
               Conv2d-82            [-1, 512, 8, 8]       2,359,808
          BatchNorm2d-83            [-1, 512, 8, 8]           1,024
                 ReLU-84            [-1, 512, 8, 8]               0
          double_conv-85            [-1, 512, 8, 8]               0
               gl_cat-86            [-1, 512, 8, 8]               0
      ConvTranspose2d-87          [-1, 512, 16, 16]       4,194,304
               Conv2d-88          [-1, 256, 16, 16]       2,359,552
          BatchNorm2d-89          [-1, 256, 16, 16]             512
                 ReLU-90          [-1, 256, 16, 16]               0
               Conv2d-91          [-1, 256, 16, 16]         590,080
          BatchNorm2d-92          [-1, 256, 16, 16]             512
                 ReLU-93          [-1, 256, 16, 16]               0
          double_conv-94          [-1, 256, 16, 16]               0
                   up-95          [-1, 256, 16, 16]               0
      ConvTranspose2d-96          [-1, 256, 32, 32]       1,048,576
               Conv2d-97          [-1, 128, 32, 32]         589,952
          BatchNorm2d-98          [-1, 128, 32, 32]             256
                 ReLU-99          [-1, 128, 32, 32]               0
              Conv2d-100          [-1, 128, 32, 32]         147,584
         BatchNorm2d-101          [-1, 128, 32, 32]             256
                ReLU-102          [-1, 128, 32, 32]               0
         double_conv-103          [-1, 128, 32, 32]               0
                  up-104          [-1, 128, 32, 32]               0
     ConvTranspose2d-105          [-1, 128, 64, 64]         262,144
              Conv2d-106           [-1, 64, 64, 64]         147,520
         BatchNorm2d-107           [-1, 64, 64, 64]             128
                ReLU-108           [-1, 64, 64, 64]               0
              Conv2d-109           [-1, 64, 64, 64]          36,928
         BatchNorm2d-110           [-1, 64, 64, 64]             128
                ReLU-111           [-1, 64, 64, 64]               0
         double_conv-112           [-1, 64, 64, 64]               0
                  up-113           [-1, 64, 64, 64]               0
     ConvTranspose2d-114         [-1, 64, 128, 128]          65,536
              Conv2d-115         [-1, 64, 128, 128]          73,792
         BatchNorm2d-116         [-1, 64, 128, 128]             128
                ReLU-117         [-1, 64, 128, 128]               0
              Conv2d-118         [-1, 64, 128, 128]          36,928
         BatchNorm2d-119         [-1, 64, 128, 128]             128
                ReLU-120         [-1, 64, 128, 128]               0
         double_conv-121         [-1, 64, 128, 128]               0
                  up-122         [-1, 64, 128, 128]               0
              Conv2d-123         [-1, 10, 128, 128]             650
             outconv-124         [-1, 10, 128, 128]               0
    ================================================================
    Total params: 27,479,393
    Trainable params: 27,479,393
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.19
    Forward/backward pass size (MB): 254.71
    Params size (MB): 104.83
    Estimated Total Size (MB): 359.72
    ----------------------------------------------------------------
    """
    class ConvNormReLU(nn.Module):
        def __init__(self, in_channels, out_channels,
                     padding=1, kernel_size=(3, 3),
                     stride=(1, 1), bias=False):
            super(ITMNet.ConvNormReLU, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          padding=padding, kernel_size=kernel_size,
                          stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d):
            super(ITMNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                ITMNet.ConvNormReLU(in_channels=in_ch, out_channels=in_ch,
                                    padding=1, kernel_size=(3, 3),
                                    stride=(1, 1), bias=False),
                down_conv_layer(in_channels=in_ch, out_channels=out_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x
    
    class gl_cat(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(ITMNet.gl_cat, self).__init__()
            self.conv = UNet.double_conv(in_ch, out_ch)

        def forward(self, x1, x2):
            x = torch.cat([x1, x2], dim=1)
            x = self.conv(x)
            return x

    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d):
        super(ITMNet, self).__init__(n_channels, n_classes,
                                     up_conv_layer, down_conv_layer)
        # global encoder
        self.g_down1 = self.down(3, 64) 
        self.g_down2 = self.down(64, 64) 
        self.g_down3 = self.down(64, 64) 
        self.g_down4 = self.down(64, 64) 
        self.g_down5 = self.down(64, 64) 
        self.g_out_conv = self.ConvNormReLU(64, 64, padding=0,
                                            kernel_size=(4, 4))
        self.gl_cat1 = self.gl_cat(512+64, 512)
        
    def forward(self, x):
        # local encoder
        lx1 = self.inc(x)
        lx2 = self.down1(lx1)
        lx3 = self.down2(lx2)
        lx4 = self.down3(lx3)
        lx5 = self.down4(lx4)

        # global encoder
        gx1 = F.interpolate(x, size=(128, 128),
                            mode="bilinear", align_corners=False)
        gx2 = self.g_down1(gx1)
        gx3 = self.g_down2(gx2)
        gx4 = self.g_down3(gx3)
        gx5 = self.g_down4(gx4)
        gx6 = self.g_down5(gx5)
        gx7 = self.g_out_conv(gx6)
        gx8 = F.interpolate(gx7, size=lx5.size()[2:],
                            mode="nearest")
        
        # fusing features
        x = self.gl_cat1(lx5, gx8)
        
        # decoder
        x = self.up1(x, lx4)
        x = self.up2(x, lx3)
        x = self.up3(x, lx2)
        x = self.up4(x, lx1)
        x = self.outc(x)
        return torch.sigmoid(x)

