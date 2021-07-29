import torch
import torch.nn as nn
import torch.nn.functional as F
import deepy.nn.layer as layer
from deepy.nn.model.senet import SELayer


class _UNetNd(nn.Module):
    """ U-Net
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super(_UNetNd.double_conv, self).__init__()
            self.conv = nn.Sequential(
                conv(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                normalization(out_channels),
                activation(),
                conv(in_channels=out_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                normalization(out_channels),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super(_UNetNd.inconv, self).__init__()
            self.conv = self.conv_block(in_channels, out_channels,
                                        conv, normalization, kernel_size,
                                        padding, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     down_conv, normalization,
                     conv_kernel_size=3,
                     conv_padding=1,
                     down_kernel_size=3,
                     down_padding=1,
                     activation=nn.ReLU):
            super(_UNetNd.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv(in_channels=in_channels, out_channels=in_channels,
                          padding=down_padding, kernel_size=down_kernel_size,
                          stride=2, bias=False),
                self.conv_block(in_channels, out_channels,
                                conv, normalization,
                                conv_kernel_size, conv_padding,
                                activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_channels: int, mid_channels: int, out_channels: int,
                     conv, up_conv, normalization,
                     conv_kernel_size=3, conv_padding=1,
                     up_kernel_size=4, up_padding=1,
                     activation=nn.ReLU):
            super(_UNetNd.up, self).__init__()
            self.upconv = up_conv(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=up_kernel_size, stride=2,
                                  padding=up_padding, bias=False)
            self.conv = self.conv_block(in_channels, out_channels,
                                        conv, normalization, conv_kernel_size,
                                        conv_padding, activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     kernel_size=1, padding=0, activation=nn.Identity):
            super(_UNetNd.outconv, self).__init__()
            self.conv = nn.Sequential(
                    conv(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         bias=True),
                    activation(),
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 conv, up_conv, down_conv,
                 normalization,
                 max_channels: int=512,
                 activation=nn.ReLU,
                 final_activation=nn.Identity):
        super(_UNetNd, self).__init__()
        self.conv_block = _UNetNd.double_conv
        self.depth = depth
        self.inc = _UNetNd.inconv(in_channels=in_channels,
                                  out_channels=base_channels,
                                  conv=conv,
                                  normalization=normalization,
                                  kernel_size=3,
                                  padding=1,
                                  activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                _UNetNd.down(
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
        self.up_blocks = nn.ModuleList(
            [
                _UNetNd.up(
                    in_channels=min(base_channels*(2**(i+1)), max_channels),
                    mid_channels=min(base_channels*(2**i), max_channels)*2,
                    out_channels=max(min(base_channels*(2**i), max_channels), base_channels),
                    conv=conv,
                    up_conv=up_conv,
                    normalization=normalization,
                    up_kernel_size=4,
                    up_padding=1,
                    activation=activation
                )
                for i in reversed(range(depth))
            ]
        )
        self.outc = _UNetNd.outconv(in_channels=base_channels,
                                    out_channels=out_channels,
                                    conv=conv,
                                    kernel_size=1,
                                    padding=0,
                                    activation=final_activation)

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        for l in self.down_blocks:
            skip_connections.append(x)
            x = l(x)
        for l in self.up_blocks:
            x = l(x, skip_connections.pop())
        x = self.outc(x)
        return x


class UNet1d(_UNetNd):
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 conv=nn.Conv1d, up_conv=nn.ConvTranspose1d, down_conv=nn.Conv1d,
                 normalization=nn.BatchNorm1d, activation=nn.ReLU,
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


class UNet2d(_UNetNd):
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


class _SEUNetNd(_UNetNd):
    """ SE U-Net
    """
    class se_conv(nn.Module):
        '''conv => BN => ReLU => SE'''
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1,
                     reduction=16, activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                conv(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                normalization(out_channels),
                activation(),
                SELayer(out_channels, reduction=reduction),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1,
                     reduction=16, activation=nn.ReLU):
            super().__init__()
            self.conv = _SEUNetNd.se_conv(in_channels, out_channels,
                                          conv, normalization, kernel_size,
                                          padding, reduction, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     down_conv, normalization,
                     conv_kernel_size=3, conv_padding=1,
                     down_kernel_size=3, down_padding=1,
                     reduction=16, activation=nn.ReLU):
            super().__init__()
            self.mpconv = nn.Sequential(
                down_conv(in_channels=in_channels, out_channels=in_channels,
                          padding=down_padding, kernel_size=down_kernel_size,
                          stride=2, bias=False),
                _SEUNetNd.se_conv(in_channels, out_channels,
                                  conv, normalization, conv_kernel_size,
                                  conv_padding, reduction, activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_channels: int, mid_channels: int, out_channels: int,
                     conv, up_conv, normalization,
                     conv_kernel_size=3, conv_padding=1,
                     up_kernel_size=4, up_padding=1,
                     reduction=16, activation=nn.ReLU):
            super().__init__()
            self.upconv = up_conv(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=up_kernel_size, stride=2,
                                  padding=up_padding, bias=False)
            self.conv = _SEUNetNd.se_conv(in_channels, out_channels,
                                          conv, normalization, conv_kernel_size,
                                          conv_padding, reduction, activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, conv,
                     kernel_size=1, padding=0, activation=nn.Identity):
            super().__init__()
            self.conv = nn.Sequential(
                    conv(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         bias=True),
                    activation(),
                )

        def forward(self, x):
            x = self.conv(x)
            return x

    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 conv, up_conv, down_conv,
                 normalization,
                 max_channels: int=512,
                 reduction: int=16,
                 activation=nn.ReLU,
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
        self.reduction = reduction
        self.inc = _SEUNetNd.inconv(in_channels=in_channels,
                                    out_channels=base_channels,
                                    conv=conv,
                                    normalization=normalization,
                                    kernel_size=3,
                                    padding=1,
                                    reduction=reduction,
                                    activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                _SEUNetNd.down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    out_channels=min(base_channels*(2**(i+1)), max_channels),
                    conv=conv,
                    down_conv=down_conv,
                    normalization=normalization,
                    down_kernel_size=3,
                    down_padding=1,
                    reduction=reduction,
                    activation=activation
                )
                for i in range(depth)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                _SEUNetNd.up(
                    in_channels=min(base_channels*(2**(i+1)), max_channels),
                    mid_channels=min(base_channels*(2**(i+1)), max_channels),
                    out_channels=max(min(base_channels*(2**(i+1)), max_channels)//2, base_channels),
                    conv=conv,
                    up_conv=up_conv,
                    normalization=normalization,
                    up_kernel_size=4,
                    up_padding=1,
                    reduction=reduction,
                    activation=activation
                )
                for i in reversed(range(depth))
            ]
        )
        self.outc = _SEUNetNd.outconv(in_channels=base_channels,
                                      out_channels=out_channels,
                                      conv=conv,
                                      kernel_size=1,
                                      padding=0,
                                      activation=final_activation)


class SEUNet1d(_SEUNetNd):
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 conv=nn.Conv1d, up_conv=nn.ConvTranspose1d, down_conv=nn.Conv1d,
                 normalization=nn.BatchNorm1d,
                 reduction=16,
                 activation=nn.ReLU, final_activation=nn.Identity):
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
            reduction=reduction,
            activation=activation,
            final_activation=final_activation
        )


class SEUNet2d(_SEUNetNd):
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 conv=nn.Conv2d, up_conv=nn.ConvTranspose2d, down_conv=nn.Conv2d,
                 normalization=nn.BatchNorm2d,
                 reduction=16,
                 activation=nn.ReLU, final_activation=nn.Identity):
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
            reduction=reduction,
            activation=activation,
            final_activation=final_activation
        )
