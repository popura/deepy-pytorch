import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy import layer


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
