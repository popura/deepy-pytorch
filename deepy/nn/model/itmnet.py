import torch
import torch.nn as nn
import torch.nn.functional as F

from deepy.nn.model import UNet, SEUNet



class ITMNet(UNet):
    """iTM-Net
    """
    class ConvNormReLU(nn.Module):
        """(conv => BN => ReLU)
        """
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                conv(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                normalization(out_channels),
                activation(),
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(
            self, in_channels, out_channels, conv, down_conv,
            normalization, down_kernel_size=3, down_padding=1,
            activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                ITMNet.ConvNormReLU(in_channels=in_channels,
                                    out_channels=in_channels,
                                    conv=conv,
                                    normalization=normalization,
                                    activation=activation),
                down_conv(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=down_kernel_size,
                          padding=down_padding,
                          stride=2, bias=False),
            )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    class gl_cat(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super().__init__()
            self.conv = ITMNet.double_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation)

        def forward(self, x1, x2):
            x = torch.cat([x1, x2], dim=1)
            x = self.conv(x)
            return x

    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 conv=nn.Conv2d, up_conv=nn.ConvTranspose2d, down_conv=nn.Conv2d,
                 normalization=nn.BatchNorm2d, activation=nn.ReLU,
                 final_activation=nn.Identity):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            max_channels=max_channels,
            conv=conv,
            up_conv=up_conv,
            down_conv=down_conv,
            normalization=normalization,
            activation=activation,
            final_activation=final_activation
        )

        # global encoder
        ge_layers = [
            ITMNet.down(
                in_channels=3,
                out_channels=64,
                conv=conv,
                down_conv=down_conv,
                normalization=normalization,
                down_kernel_size=3,
                down_padding=1,
                activation=activation
            )
        ]
        ge_layers.extend([
            ITMNet.down(
                in_channels=64,
                out_channels=64,
                conv=conv,
                down_conv=down_conv,
                normalization=normalization,
                down_kernel_size=3,
                down_padding=1,
                activation=activation
            )
            for i in reversed(range(4))
        ])
        ge_layers.append(
            ITMNet.ConvNormReLU(
                in_channels=64,
                out_channels=64,
                conv=conv,
                normalization=normalization,
                kernel_size=4,
                padding=0,
                activation=activation
            )
        )
        self.g_down = nn.Sequential(*list(ge_layers))
        self.cat = self.gl_cat(
            in_channels=512+64,
            out_channels=512,
            conv=conv,
            normalization=normalization)
        
    def forward(self, x):
        skip_connections = []

        # local encoder
        lx = self.inc(x)
        for l in self.down_blocks:
            skip_connections.append(lx)
            lx = l(lx)
        
        # global encoder
        gx = F.interpolate(x, size=(128, 128),
                           mode="bilinear", align_corners=False)
        gx = self.g_down(gx)
        gx = F.interpolate(gx, size=lx.size()[2:],
                           mode="nearest")
        
        # concat feature maps
        x = self.cat(lx, gx)
    
        # decoder
        for l in self.up_blocks:
            x = l(x, skip_connections.pop())
        x = self.outc(x)
        return x


class SEITMNet(SEUNet):
    class ConvNormAct(nn.Module):
        def __init__(self, in_channels, out_channels,
                     padding=1, kernel_size=(3, 3),
                     stride=(1, 1), bias=False,
                     activation=nn.ReLU):
            super(SEITMNet.ConvNormAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          padding=padding, kernel_size=kernel_size,
                          stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super(SEITMNet.down, self).__init__()
            self.mpconv = nn.Sequential(
                SEITMNet.ConvNormAct(in_channels=in_ch, out_channels=in_ch,
                                   padding=1, kernel_size=(3, 3),
                                   stride=(1, 1), bias=False,
                                   activation=activation),
                down_conv_layer(in_channels=in_ch, out_channels=out_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x
    
    class gl_cat(nn.Module):
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(SEITMNet.gl_cat, self).__init__()
            self.conv = SEUNet.se_conv(in_ch, out_ch, activation=activation)

        def forward(self, x1, x2):
            x = torch.cat([x1, x2], dim=1)
            x = self.conv(x)
            return x

    def __init__(self, n_channels, n_classes,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(SEITMNet, self).__init__(n_channels, n_classes,
                                     up_conv_layer, down_conv_layer,
                                     activation=activation)
        # global encoder
        self.g_down1 = self.down(3, 64, activation=activation) 
        self.g_down2 = self.down(64, 64, activation=activation) 
        self.g_down3 = self.down(64, 64, activation=activation) 
        self.g_down4 = self.down(64, 64, activation=activation) 
        self.g_down5 = self.down(64, 64, activation=activation) 
        self.g_out_conv = self.ConvNormAct(64, 64, padding=0,
                                           kernel_size=(4, 4),
                                           activation=activation)
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
        return x
