import copy
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy.nn.layer import InvertibleModule, Split, Join, Lift, Drop


class _InvertibleUNetNd(InvertibleModule):
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_channels: int, out_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super(_InvertibleUNetNd.double_conv, self).__init__()
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

    class lift_conv(InvertibleModule):
        def __init__(self, in_channels: int, conv, normalization,
                     kernel_size=3, padding=1, activation=nn.ReLU):
            super(_InvertibleUNetNd.lift_conv, self).__init__()
            self.f = _InvertibleUNetNd.double_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            self.g = _InvertibleUNetNd.double_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            self.lift = Lift(self.f, self.g)
        
        def forward(self, x1, x2):
            return self.lift.forward(x1, x2)
        
        def rearward(self, y1, y2):
            return self.lift.rearward(y1, y2)

    class drop_conv(InvertibleModule):
        def __init__(self, in_channels: int, conv, normalization,
                     kernel_size=3, padding=1, activation=nn.ReLU):
            super(_InvertibleUNetNd.drop_conv, self).__init__()
            self.f = _InvertibleUNetNd.double_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            self.g = _InvertibleUNetNd.double_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            self.drop = Drop(self.f, self.g)
        
        def forward(self, x1, x2):
            return self.drop.forward(x1, x2)
        
        def rearward(self, y1, y2):
            return self.drop.rearward(y1, y2)

    class inconv(InvertibleModule):
        def __init__(self, in_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super(_InvertibleUNetNd.inconv, self).__init__()
            self.f = _InvertibleUNetNd.double_conv(
                in_channels=in_channels//2,
                out_channels=in_channels//2,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            self.g = _InvertibleUNetNd.double_conv(
                in_channels=in_channels//2,
                out_channels=in_channels//2,
                conv=conv,
                normalization=normalization,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            self.lift = Lift(self.f, self.g)
            self.in_channels = in_channels
        
        def forward(self, x):
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = self.lift(x1, x2)
            return torch.cat([x1, x2], dim=1)
            
        def rearward(self, y):
            x1, x2 = torch.chunk(y, 2, dim=1)
            x1, x2 = self.lift.rearward(x1, x2)
            return torch.cat([x1, x2], dim=1)

    class outconv(InvertibleModule):
        def __init__(self, in_channels: int, conv,
                     normalization, kernel_size=3, padding=1, activation=nn.ReLU):
            super(InvertibleUNet.outconv, self).__init__()
            self.f = nn.Sequential(
                conv(in_channels=in_channels//2,
                     out_channels=in_channels//2,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                activation()
            )
            self.g = nn.Sequential(
                conv(in_channels=in_channels//2,
                     out_channels=in_channels//2,
                     kernel_size=kernel_size,
                     padding=padding,
                     bias=False),
                activation()
            )
            self.lift = Lift(self.f, self.g)
        
        def forward(self, x):
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = self.lift.forward(x1, x2)
            return torch.cat([x1, x2], dim=1)
        
        def rearward(self, y):
            x1, x2 = torch.chunk(y, 2, dim=1)
            x1, x2 = self.lift.rearward(x1, x2)
            return torch.cat([x1, x2], dim=1)

    class _split(InvertibleModule):
        def __init__(self, dim):
            super(_InvertibleUNetNd._split, self).__init__()
            self.splitters = nn.ModuleList([Split(dim=-i) for i in range(1, dim+1)])

        def forward(self, x):
            in_list = [x]
            for s in self.splitters:
                out_list = []
                for i in in_list:
                    out_list.extend(s(i))
                in_list = copy.copy(out_list)

            return tuple(out_list)
        
        def rearward(self, *args):
            in_list = list(args)
            for s in reversed(self.splitters):
                out_list = []
                for i in range(0, len(in_list), 2):
                    out_list.append(s.rearward(in_list[i], in_list[i+1]))
                in_list = copy.copy(out_list)
            return out_list[0]

    class _join(InvertibleModule):
        def __init__(self, dim):
            super(_InvertibleUNetNd._join, self).__init__()
            self.joiners = nn.ModuleList([Join(dim=-i) for i in range(1, dim+1)])

        def forward(self, *args):
            in_list = list(args)
            for j in reversed(self.joiners):
                out_list = []
                for i in range(0, len(in_list), 2):
                    out_list.append(j(in_list[i], in_list[i+1]))
                in_list = copy.copy(out_list)
            return out_list[0]

        def rearward(self, x):
            in_list = [x]
            for j in self.joiners:
                out_list = []
                for i in in_list:
                    out_list.extend(j.rearward(i))
                in_list = copy.copy(out_list)

            return tuple(out_list)
        
    class down(InvertibleModule):
        def __init__(self, in_channels, dim, conv, normalization,
                     kernel_size=3, padding=1, activation=nn.ReLU):
            super(_InvertibleUNetNd.down, self).__init__()
            self.split = _InvertibleUNetNd._split(dim=dim)
            self.conv = _InvertibleUNetNd.lift_conv(in_channels=in_channels*2,
                                                    conv=conv,
                                                    normalization=normalization,
                                                    kernel_size=kernel_size,
                                                    padding=padding,
                                                    activation=activation) 

        def forward(self, x):
            # TODO: fix here
            x_list = list(self.split(x))
            x1 = torch.cat(x_list[:len(x_list)//2], dim=1)
            x2 = torch.cat(x_list[len(x_list)//2:], dim=1)
            x1, x2 = self.conv(x1, x2)
            return x1, x2
        
        def rearward(self, x1, x2):
            # TODO: fix here
            x1, x2 = self.conv.rearward(x1, x2)
            xx, xy = torch.chunk(x1, 2, dim=1)
            yx, yy = torch.chunk(x2, 2, dim=1)
            x = self.split.rearward(xx, xy, yx, yy)
            return x

    class up(InvertibleModule):
        def __init__(self, in_channels, activation=nn.ReLU):
            super(_InvertibleUNetNd.up, self).__init__()
            self.join = _InvertibleUNetNd._join()
            self.conv = _InvertibleUNetNd.lift_conv(in_channels=in_channels*2,
                                                    conv=conv,
                                                    normalization=normalization,
                                                    kernel_size=kernel_size,
                                                    padding=padding,
                                                    activation=activation) 
        def forward(self, x1, x2):
            # TODO: fix here
            x1, x2 = self.conv(x1, x2)
            xx, xy = torch.chunk(x1, 2, dim=1)
            yx, yy = torch.chunk(x2, 2, dim=1)
            x = self.join(xx, xy, yx, yy)
            return x

        def rearward(self, x):
            # TODO: fix here
            xx, xy, yx, yy = self.join.rearward(x)
            x1 = torch.cat([xx, xy], dim=1)
            x2 = torch.cat([yx, yy], dim=1)
            x1, x2 = self.conv.rearward(x1, x2)
            return x1, x2

    def __init__(self, in_channels: int, dim: int, depth: int,
                 conv, up_conv, down_conv,
                 normalization,
                 max_channels: int=512,
                 activation=nn.ReLU,
                 final_activation=nn.Identity):
        super(_InvertibleUNetNd, self).__init__()
        if in_channels % 2 == 1:
            raise ValueError("n_channels should be a multiple of 2")
        self.base_channels = in_channels
        self.depth = depth
        self.inc = _InvertibleUNetNd.inconv(
            in_channels=in_channels,
            conv=conv,
            normalization=normalization,
            kernel_size=3,
            padding=1,
            activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                _InvertibleUNetNd.down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    conv=conv,
                    down_conv=down_conv,
                    normalization=normalization,
                    kernel_size=3,
                    padding=1,
                    activation=activation
                )
                for i in range(depth)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                _InvertibleUNetNd.up(
                    in_channels=min(base_channels*(2**(i+1)), max_channels),
                    conv=conv,
                    up_conv=up_conv,
                    normalization=normalization,
                    kernel_size=4,
                    padding=1,
                    activation=activation
                )
                for i in reversed(range(depth))
            ]
        )
        self.outc = _InvertibleUNetNd.outconv(
            in_channels=base_channels,
            out_channels=out_channels,
            conv=conv,
            kernel_size=1,
            padding=0,
            activation=final_activation)

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        for l in self.down_blocks:
            x, y = l(x)
            skip_connections.append(y)
        for l in self.up_blocks:
            x = l(x, skip_connections.pop())
        x = self.outc(x)
        return x
    
    def rearward(self, x):
        skip_connections = []
        x = self.outc.rearward(x)
        for l in self.up_blocks:
            x, y = l.rearward(x)
            skip_connections.append(y)
        for l in self.up_blocks:
            x = l.rearward(x, skip_connections.pop())
        x = self.inc.rearward(x)
        return x


class InvertibleUNet(InvertibleModule):
    class lift_conv(InvertibleModule):
        def __init__(self, n_channels, activation=nn.ReLU):
            super(InvertibleUNet.lift_conv, self).__init__()
            self.f = nn.Sequential(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(n_channels),
                activation())
            self.g = nn.Sequential(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(n_channels),
                activation()
            )
            self.lift = Lift(self.f, self.g)
        
        def forward(self, x1, x2):
            return self.lift.forward(x1, x2)
        
        def rearward(self, y1, y2):
            return self.lift.rearward(y1, y2)

    class drop_conv(InvertibleModule):
        def __init__(self, n_channels, activation=nn.ReLU):
            super(InvertibleUNet.drop_conv, self).__init__()
            self.f = nn.Sequential(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(n_channels),
                activation())
            self.g = nn.Sequential(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(n_channels),
                activation()
            )
            self.drop = Drop(self.f, self.g)
        
        def forward(self, x1, x2):
            return self.drop.forward(x1, x2)
        
        def rearward(self, y1, y2):
            return self.drop.rearward(y1, y2)

    class xy_split(InvertibleModule):
        def __init__(self, ):
            super(InvertibleUNet.xy_split, self).__init__()
            self.col_split = Split(dim=-1)
            self.row_split = Split(dim=-2)

        def forward(self, x):
            x1, x2 = self.col_split(x)
            xx, xy = self.row_split(x1)
            yx, yy = self.row_split(x2)

            return xx, xy, yx, yy
        
        def rearward(self, xx, xy, yx, yy):
            x2 = self.row_split.rearward(yx, yy)
            x1 = self.row_split.rearward(xx, xy)
            x = self.col_split.rearward(x1, x2)
            return x

    class xy_join(InvertibleModule):
        def __init__(self):
            super(InvertibleUNet.xy_join, self).__init__()
            self.col_join = Join(dim=-1)
            self.row_join = Join(dim=-2)
        
        def forward(self, xx, xy, yx, yy):
            x1 = self.row_join(xx, xy)
            x2 = self.row_join(yx, yy)
            x = self.col_join(x1, x2)
            return x

        def rearward(self, y):
            x1, x2 = self.col_join.rearward(y)
            yx, yy = self.row_join.rearward(x2)
            xx, xy = self.row_join.rearward(x1)
            return xx, xy, yx, yy
    
    class down(InvertibleModule):
        def __init__(self, in_channels, activation=nn.ReLU):
            super(InvertibleUNet.down, self).__init__()
            self.split = InvertibleUNet.xy_split()
            self.conv = InvertibleUNet.lift_conv(n_channels=in_channels*2,
                                                 activation=activation) 

        def forward(self, x):
            xx, xy, yx, yy = self.split(x)
            x1 = torch.cat([xx, xy], dim=1)
            x2 = torch.cat([yx, yy], dim=1)
            x1, x2 = self.conv(x1, x2)
            return x1, x2
        
        def rearward(self, x1, x2):
            x1, x2 = self.conv.rearward(x1, x2)
            xx, xy = torch.chunk(x1, 2, dim=1)
            yx, yy = torch.chunk(x2, 2, dim=1)
            x = self.split.rearward(xx, xy, yx, yy)
            return x

    class up(InvertibleModule):
        def __init__(self, in_channels, activation=nn.ReLU):
            super(InvertibleUNet.up, self).__init__()
            self.join = InvertibleUNet.xy_join()
            self.conv = InvertibleUNet.lift_conv(n_channels=in_channels,
                                                 activation=activation) 

        def forward(self, x1, x2):
            x1, x2 = self.conv(x1, x2)
            xx, xy = torch.chunk(x1, 2, dim=1)
            yx, yy = torch.chunk(x2, 2, dim=1)
            x = self.join(xx, xy, yx, yy)
            return x

        def rearward(self, x):
            xx, xy, yx, yy = self.join.rearward(x)
            x1 = torch.cat([xx, xy], dim=1)
            x2 = torch.cat([yx, yy], dim=1)
            x1, x2 = self.conv.rearward(x1, x2)
            return x1, x2

    class inconv(InvertibleModule):
        def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
            super(InvertibleUNet.inconv, self).__init__()
            self.f = nn.Sequential(
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(out_channels//2),
                activation())
            self.g = nn.Sequential(
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(out_channels//2),
                activation()
            )
            self.lift = Lift(self.f, self.g)
            self.in_channels = in_channels
            self.out_channels = out_channels
        
        def forward(self, x):
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = self.lift(x1, x2)
            return torch.cat([x1, x2], dim=1)
            
        def rearward(self, y):
            x1, x2 = torch.chunk(y, 2, dim=1)
            x1, x2 = self.lift.rearward(x1, x2)
            return torch.cat([x1, x2], dim=1)

    class outconv(InvertibleModule):
        def __init__(self, in_channels, activation=nn.ReLU()):
            super(InvertibleUNet.outconv, self).__init__()
            self.f = nn.Sequential(
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                activation())
            self.g = nn.Sequential(
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2,
                          kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),
                          bias=False),
                activation()
            )
            self.lift = Lift(self.f, self.g)
        
        def forward(self, x):
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = self.lift.forward(x1, x2)
            return torch.cat([x1, x2], dim=1)
        
        def rearward(self, y):
            x1, x2 = torch.chunk(y, 2, dim=1)
            x1, x2 = self.lift.rearward(x1, x2)
            return torch.cat([x1, x2], dim=1)
    
    def __init__(self, n_channels, activation=nn.ReLU):
        super(InvertibleUNet, self).__init__()
        if n_channels % 2 == 1:
            raise ValueError("n_channels should be a multiple of 2")
        base_channels = n_channels
        self.inc = InvertibleUNet.inconv(n_channels, base_channels, activation=activation)
        self.down1 = InvertibleUNet.down(base_channels, activation=activation)
        self.down2 = InvertibleUNet.down((2**1)*base_channels, activation=activation)
        self.down3 = InvertibleUNet.down((2**2)*base_channels, activation=activation)
        self.down4 = InvertibleUNet.down((2**3)*base_channels, activation=activation)
        self.up1 = InvertibleUNet.up((2**4)*base_channels, activation=activation)
        self.up2 = InvertibleUNet.up((2**3)*base_channels, activation=activation)
        self.up3 = InvertibleUNet.up((2**2)*base_channels, activation=activation)
        self.up4 = InvertibleUNet.up((2**1)*base_channels, activation=activation)
        self.outc = InvertibleUNet.outconv(base_channels, activation=activation)

    def forward(self, x):
        x1 = self.inc(x)
        x2, y2 = self.down1(x1)
        x3, y3 = self.down2(x2)
        x4, y4 = self.down3(x3)
        x5, y5 = self.down4(x4)
        x = self.up1(x5, y5)
        x = self.up2(x, y4)
        x = self.up3(x, y3)
        x = self.up4(x, y2)
        x = self.outc(x)
        return x
    
    def rearward(self, y):
        x1 = self.outc.rearward(y)
        x2, y2 = self.up4.rearward(x1)
        x3, y3 = self.up3.rearward(x2)
        x4, y4 = self.up2.rearward(x3)
        x5, y5 = self.up1.rearward(x4)
        x4 = self.down4.rearward(x5, y5)
        x3 = self.down3.rearward(x4, y4)
        x2 = self.down2.rearward(x3, y3)
        x1 = self.down1.rearward(x2, y2)
        x = self.inc.rearward(x1)
        return x