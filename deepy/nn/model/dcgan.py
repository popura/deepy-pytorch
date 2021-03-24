import torch
import torch.nn as nn
import torch.nn.functional as F
from deepy import layer


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, trans_conv_layer=nn.ConvTranspose2d):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            trans_conv_layer(in_channels=latent_dim, out_channels=latent_dim*8,
                             kernel_size=(4, 4), stride=(1, 1),
                             padding=0, bias=False),
            nn.BatchNorm2d(num_features=latent_dim*8),
            nn.ReLU(inplace=True),

            trans_conv_layer(in_channels=latent_dim*8, out_channels=latent_dim*4,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=latent_dim*4),
            nn.ReLU(inplace=True),

            trans_conv_layer(in_channels=latent_dim*4, out_channels=latent_dim*2,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=latent_dim*2),
            nn.ReLU(inplace=True),

            trans_conv_layer(in_channels=latent_dim*2, out_channels=latent_dim,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.BatchNorm2d(num_features=latent_dim),
            nn.ReLU(inplace=True),

            trans_conv_layer(in_channels=latent_dim, out_channels=3,
                             kernel_size=(4, 4), stride=(2, 2),
                             padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), self.latent_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    def __init__(self, down_conv_layer=nn.Conv2d, normalization="BN", n_fmap=64):
        super(DCGANDiscriminator, self).__init__()
        if normalization == "BN":
            norm = nn.BatchNorm2d
        else:
            raise NotImplementedError()
        self.n_fmap=64

        self.net = nn.Sequential(
            down_conv_layer(in_channels=3, out_channels=n_fmap,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            down_conv_layer(in_channels=n_fmap, out_channels=n_fmap*2,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=n_fmap*2, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            down_conv_layer(in_channels=n_fmap*2, out_channels=n_fmap*4,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=n_fmap*4, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            down_conv_layer(in_channels=n_fmap*4, out_channels=n_fmap*8,
                            padding=1, kernel_size=(4, 4),
                            stride=(2, 2), bias=False),
            norm(num_features=n_fmap*8, affine=True,
                 track_running_stats=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Conv2d(in_channels=n_fmap*8, out_channels=1,
                      padding=0, kernel_size=(4, 4),
                      stride=(1, 1), bias=False),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))
