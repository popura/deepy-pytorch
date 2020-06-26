import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy import layer


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