import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import deepy.nn.layer as layer


class OriginalVGG(nn.Module):
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
        super().__init__()
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


class ConvNormAct(nn.Module):
    """ This module applies
        conv => normalization => activation
        multiple times.
    """
    def __init__(self, in_channels: int, out_channels: int, conv,
                 normalization, kernel_size: int = 3, padding: int = 1, activation=nn.ReLU,
                 times: int = 2):
        super().__init__()
        self.times = times
        layers = [
            conv(in_channels=in_channels,
                 out_channels=out_channels,
                 kernel_size=kernel_size,
                 padding=padding,
                 bias=False),
            normalization(out_channels),
            activation()]
        for i in range(self.times - 1):
            layers.extend([
                conv(in_channels=out_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                normalization(out_channels),
                activation()])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv,
                 down_conv, normalization,
                 conv_kernel_size=3,
                 conv_padding=1,
                 down_kernel_size=3,
                 down_padding=1,
                 activation=nn.ReLU):
        super().__init__()
        self.mpconv = nn.Sequential(
            down_conv(in_channels=in_channels, out_channels=in_channels,
                      padding=down_padding, kernel_size=down_kernel_size,
                      stride=2, bias=False),
            ConvNormAct(in_channels, out_channels,
                        conv, normalization,
                        conv_kernel_size, conv_padding,
                        activation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class _VGGNd(nn.Module):
    """ _VGGNd
    """

    def __init__(self, in_channels: int, num_classes: int,
                 base_channels: int, depth: int,
                 conv, down_conv,
                 normalization,
                 max_channels: int=512,
                 activation=nn.ReLU):
        super().__init__()
        self.depth = depth
        self.inc = ConvNormAct(in_channels=in_channels,
                               out_channels=base_channels,
                               conv=conv,
                               normalization=normalization,
                               kernel_size=3,
                               padding=1,
                               activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                Down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    out_channels=min(base_channels*(2**(i+1)), max_channels),
                    conv=conv,
                    down_conv=down_conv,
                    normalization=normalization,
                    down_kernel_size=3,
                    down_padding=1,
                    activation=activation
                )
                for i in range(depth)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(
            nn.Linear(min(base_channels*(2**depth), max_channels), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.inc(x)
        for i, l in enumerate(self.down_blocks):
            x = l(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class VGG1d(_VGGNd):
    def __init__(self, in_channels: int, num_classes: int,
                 base_channels: int, depth: int,
                 conv=nn.Conv1d, down_conv=nn.Conv1d,
                 normalization=nn.BatchNorm1d,
                 max_channels: int=512,
                 activation=nn.ReLU):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            conv=conv,
            down_conv=down_conv,
            normalization=normalization,
            max_channels=max_channels,
            activation=activation,
        )


class VGG2d(_VGGNd):
    def __init__(self, in_channels: int, num_classes: int,
                 base_channels: int, depth: int,
                 conv=nn.Conv2d, down_conv=nn.Conv2d,
                 normalization=nn.BatchNorm2d,
                 max_channels: int=512,
                 activation=nn.ReLU):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            conv=conv,
            down_conv=down_conv,
            normalization=normalization,
            max_channels=max_channels,
            activation=activation,
        )