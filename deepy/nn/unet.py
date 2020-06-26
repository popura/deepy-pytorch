import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy import layer


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
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(UNet.double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(UNet.inconv, self).__init__()
            self.conv = UNet.double_conv(in_ch, out_ch, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super(UNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
                UNet.double_conv(in_ch, out_ch, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, up_conv_layer=nn.ConvTranspose2d, activation=nn.ReLU):
            super(UNet.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                        kernel_size=(4, 4), stride=(2, 2),
                                        padding=1, bias=False)
            self.conv = UNet.double_conv(mid_ch, out_ch, activation=activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet.outconv, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.ReLU(inplace=True)
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(UNet, self).__init__()
        self.inc = UNet.inconv(n_channels, 64, activation=activation)
        self.down1 = UNet.down(64, 128, activation=activation)
        self.down2 = UNet.down(128, 256, activation=activation)
        self.down3 = UNet.down(256, 512, activation=activation)
        self.down4 = UNet.down(512, 512, activation=activation)
        self.up1 = UNet.up(512, 1024, 256, activation=activation)
        self.up2 = UNet.up(256, 512, 128, activation=activation)
        self.up3 = UNet.up(128, 256, 64, activation=activation)
        self.up4 = UNet.up(64, 128, 64, activation=activation)
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
        return x


class SEUNet(UNet):
    """ U-Net
    """
    class se_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(SEUNet.se_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation(),
                SELayer(out_ch, reduction=16),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(SEUNet.inconv, self).__init__()
            self.conv = SEUNet.se_conv(in_ch, out_ch, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super(SEUNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
                SEUNet.se_conv(in_ch, out_ch, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, up_conv_layer=nn.ConvTranspose2d, activation=nn.ReLU):
            super(SEUNet.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                        kernel_size=(4, 4), stride=(2, 2),
                                        padding=1, bias=False)
            self.conv = SEUNet.se_conv(mid_ch, out_ch, activation=activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(SEUNet.outconv, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.ReLU(inplace=True)
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(SEUNet, self).__init__(n_channels, n_classes)
        self.inc = SEUNet.inconv(n_channels, 64, activation=activation)
        self.down1 = SEUNet.down(64, 128, activation=activation)
        self.down2 = SEUNet.down(128, 256, activation=activation)
        self.down3 = SEUNet.down(256, 512, activation=activation)
        self.down4 = SEUNet.down(512, 512, activation=activation)
        self.up1 = SEUNet.up(512, 1024, 256, activation=activation)
        self.up2 = SEUNet.up(256, 512, 128, activation=activation)
        self.up3 = SEUNet.up(128, 256, 64, activation=activation)
        self.up4 = SEUNet.up(64, 128, 64, activation=activation)
        self.outc = SEUNet.outconv(64, n_classes)

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
        return x