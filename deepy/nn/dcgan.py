import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy import layer


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