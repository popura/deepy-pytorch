import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy import layer


class FullyConnectedAutoEncoder(nn.Module):
    """
        FullyConnectedAutoEncoder
    """

    def __init__(self, in_features, n_features):
        super(FullyConnectedAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, n_features),
                nn.BatchNorm1d(n_features),
                nn.ReLU(inplace=True)
            )
        
        self.bottom = nn.Sequential(
                nn.Linear(n_features, n_features),
                nn.Tanh()
            )
        
        self.decoder = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                        
                nn.Linear(128, in_features)
            )

    def forward(self, x):
        x_size = x.size()
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x)
        x = x.view(x_size)
        return x


class ResidualFullyConnectedAutoEncoder(FullyConnectedAutoEncoder):
    """
        FullyConnectedAutoEncoder
    """

    class ResidualBlock(nn.Module):
        def __init__(self, in_features, out_features):
            super(ResidualFullyConnectedAutoEncoder.ResidualBlock, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.linear = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(inplace=True)
                )
        
        def forward(self, x):
            return self.linear(x) + x

    def __init__(self, in_features, n_features):
        super(ResidualFullyConnectedAutoEncoder, self).__init__(in_features, n_features)

        self.encoder = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                ResidualFullyConnectedAutoEncoder.ResidualBlock(128, 128),

                ResidualFullyConnectedAutoEncoder.ResidualBlock(128, 128),

                ResidualFullyConnectedAutoEncoder.ResidualBlock(128, 128),

                nn.Linear(128, n_features),
                nn.BatchNorm1d(n_features),
                nn.ReLU(inplace=True)
            )
        
        self.decoder = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                ResidualFullyConnectedAutoEncoder.ResidualBlock(128, 128),

                ResidualFullyConnectedAutoEncoder.ResidualBlock(128, 128),

                ResidualFullyConnectedAutoEncoder.ResidualBlock(128, 128),

                nn.Linear(128, in_features)
            )


class ConvAutoEncoder(nn.Module):
    """
        AutoEncoder
    """

    def __init__(self, in_channels, out_channels, n_features):
        super(ConvAutoEncoder, self).__init__()
        self.n_features = n_features

        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=n_features,
                                  kernel_size=(3, 11), stride=(2, 5),
                                  padding=(1, 5), bias=False),
                        nn.BatchNorm2d(n_features),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels=n_features, out_channels=n_features,
                                  kernel_size=(3, 11), stride=(2, 5),
                                  padding=(1, 5), bias=False),
                        nn.BatchNorm2d(n_features),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels=n_features, out_channels=n_features,
                                  kernel_size=(3, 11), stride=(2, 5),
                                  padding=(1, 5), bias=False),
                        nn.BatchNorm2d(n_features),
                        nn.ReLU(inplace=True),
                    )
        self.bottom = nn.Sequential(
                        nn.Conv2d(in_channels=n_features, out_channels=out_channels,
                                  kernel_size=(3, 3), stride=(1, 1),
                                  padding=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_features,
                                           kernel_size=(4, 11), stride=(2, 5),
                                           padding=(1, 3), bias=False),
                        nn.BatchNorm2d(n_features),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(in_channels=n_features, out_channels=n_features,
                                           kernel_size=(4, 11), stride=(2, 5),
                                           padding=(1, 3), bias=False),
                        nn.BatchNorm2d(n_features),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(in_channels=n_features, out_channels=in_channels,
                                           kernel_size=(4, 11), stride=(2, 5),
                                           padding=(1, 3), bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x)
        return x