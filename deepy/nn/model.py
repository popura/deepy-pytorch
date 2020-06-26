import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepy import layer


class DCGANGeneratorCIFER10(nn.Module):
    """Generator of DCGAN for CIFER10
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = DCGANGeneratorCIFER10(10).to(device)
    >>> summary(net , (10, ))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                 [-1, 8192]          90,112
       ConvTranspose2d-2            [-1, 256, 8, 8]       2,097,408
           BatchNorm2d-3            [-1, 256, 8, 8]             512
                  ReLU-4            [-1, 256, 8, 8]               0
          UpConvBNReLU-5            [-1, 256, 8, 8]               0
       ConvTranspose2d-6          [-1, 128, 16, 16]         524,416
           BatchNorm2d-7          [-1, 128, 16, 16]             256
                  ReLU-8          [-1, 128, 16, 16]               0
          UpConvBNReLU-9          [-1, 128, 16, 16]               0
      ConvTranspose2d-10           [-1, 64, 32, 32]         131,136
          BatchNorm2d-11           [-1, 64, 32, 32]             128
                 ReLU-12           [-1, 64, 32, 32]               0
         UpConvBNReLU-13           [-1, 64, 32, 32]               0
               Conv2d-14            [-1, 3, 32, 32]           1,731
                 Tanh-15            [-1, 3, 32, 32]               0
    ================================================================
    Total params: 2,845,699
    Trainable params: 2,845,699
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 3.61
    Params size (MB): 10.86
    Estimated Total Size (MB): 14.46
    ----------------------------------------------------------------
    """
    class UpConvBNReLU(nn.Module):
        def __init__(self, in_channels, out_channels,
                     kernel_size, stride, padding=0,
                     num_classes=0, up_sampling_layer=nn.ConvTranspose2d):
            super(DCGANGeneratorCIFER10.UpConvBNReLU, self).__init__()
            self.up = up_sampling_layer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding)

            if num_classes == 0:
                self.bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.bn = layer.ConditionalBatchNorm2d(
                            num_features=out_channels,
                            num_classes=num_classes)

            self.relu = nn.ReLU(inplace=True)

        def forward(self, inputs, label_onehots=None):
            x = self.up(inputs)
            if label_onehots is not None:
                x = self.bn(x, label_onehots)
            else:
                x = self.bn(x)
            return self.relu(x)

    def __init__(self, latent_dim, num_classes=0,
                 output_shape=(3, 32, 32),
                 up_sampling_layer=nn.ConvTranspose2d):
        super(DCGANGeneratorCIFER10, self).__init__()
        self.latent_dim = latent_dim
        self.mg = output_shape[1] // 8
        self.dense = nn.Linear(latent_dim, self.mg*self.mg*512)
        self.conv1 = DCGANGeneratorCIFER10.UpConvBNReLU(
                       in_channels=512, out_channels=256,
                       kernel_size=(4, 4), stride=(2, 2),
                       padding=1, num_classes=num_classes,
                       up_sampling_layer=up_sampling_layer)
        self.conv2 = DCGANGeneratorCIFER10.UpConvBNReLU(
                       in_channels=256, out_channels=128,
                       kernel_size=(4, 4), stride=(2, 2),
                       padding=1, num_classes=num_classes,
                       up_sampling_layer=up_sampling_layer)
        self.conv3 = DCGANGeneratorCIFER10.UpConvBNReLU(
                       in_channels=128, out_channels=64,
                       kernel_size=(4, 4), stride=(2, 2),
                       padding=1, num_classes=num_classes,
                       up_sampling_layer=up_sampling_layer)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, inputs, label_onehots=None):
        x = inputs.view(inputs.size(0), self.latent_dim)
        x = self.dense(x).view(inputs.size(0), 512, self.mg, self.mg)
        x = self.conv1(x, label_onehots)
        x = self.conv2(x, label_onehots)
        x = self.conv3(x, label_onehots)
        return self.out(x)


class DCGANDiscriminatorCIFER10(nn.Module):
    """Discriminator of DCGAN for CIFER10
    
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = DCGANDiscriminatorCIFER10().to(device)
    >>> summary(net , (3, 32, 32))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
          SpectralNorm-1           [-1, 64, 32, 32]               0
             LeakyReLU-2           [-1, 64, 32, 32]               0
          ConvNormReLU-3           [-1, 64, 32, 32]               0
          SpectralNorm-4           [-1, 64, 16, 16]               0
             LeakyReLU-5           [-1, 64, 16, 16]               0
          ConvNormReLU-6           [-1, 64, 16, 16]               0
          SpectralNorm-7          [-1, 128, 16, 16]               0
             LeakyReLU-8          [-1, 128, 16, 16]               0
          ConvNormReLU-9          [-1, 128, 16, 16]               0
         SpectralNorm-10            [-1, 128, 8, 8]               0
            LeakyReLU-11            [-1, 128, 8, 8]               0
         ConvNormReLU-12            [-1, 128, 8, 8]               0
         SpectralNorm-13            [-1, 256, 8, 8]               0
            LeakyReLU-14            [-1, 256, 8, 8]               0
         ConvNormReLU-15            [-1, 256, 8, 8]               0
         SpectralNorm-16            [-1, 256, 4, 4]               0
            LeakyReLU-17            [-1, 256, 4, 4]               0
         ConvNormReLU-18            [-1, 256, 4, 4]               0
         SpectralNorm-19            [-1, 512, 4, 4]               0
            LeakyReLU-20            [-1, 512, 4, 4]               0
         ConvNormReLU-21            [-1, 512, 4, 4]               0
               Linear-22                    [-1, 1]           8,193
    ================================================================
    Total params: 8,193
    Trainable params: 8,193
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 3.47
    Params size (MB): 0.03
    Estimated Total Size (MB): 3.51
    ----------------------------------------------------------------
    """
    class ConvNormReLU(nn.Module):
        def __init__(self, in_channels, out_channels,
                     kernel_size, stride, padding=0,
                     normalization="BN",
                     lrelu_slope=0.1, down_sampling_layer=nn.Conv2d):
            super(DCGANDiscriminatorCIFER10.ConvNormReLU, self).__init__()

            self.conv = down_sampling_layer(
                          in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding)
            if normalization == "SN":
                self.norm = layer.SpectralNorm(self.conv)
            elif normalization == "BN":
                self.norm = nn.Sequential(
                    self.conv,
                    nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError()

            self.lrelu = nn.LeakyReLU(lrelu_slope, inplace=True)

        def forward(self, inputs):
            return self.lrelu(self.norm(inputs))

    def __init__(self, num_classes=0,
                 input_shape=(3, 32, 32),
                 normalization="SN",
                 down_sampling_layer=nn.Conv2d):
        super(DCGANDiscriminatorCIFER10, self).__init__()
        self.mg = input_shape[1] // 8

        self.conv1 = self.discriminator_block(
            in_channels=3, out_channels=64,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.conv2 = self.discriminator_block(
            in_channels=64, out_channels=128,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.conv3 = self.discriminator_block(
            in_channels=128, out_channels=256,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.conv4 = DCGANDiscriminatorCIFER10.ConvNormReLU(
            in_channels=256, out_channels=512,
            kernel_size=(3, 3), stride=(1, 1), padding=1,
            normalization=normalization,
            down_sampling_layer=down_sampling_layer)
        self.dense = nn.Linear(self.mg * self.mg * 512, 1)

    def discriminator_block(self, in_channels, out_channels,
                            normalization, down_sampling_layer):
        return nn.Sequential(
            DCGANDiscriminatorCIFER10.ConvNormReLU(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(3, 3), stride=(1, 1), padding=1,
                normalization=normalization,
                down_sampling_layer=down_sampling_layer),
            DCGANDiscriminatorCIFER10.ConvNormReLU(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=(4, 4), stride=(2, 2), padding=1,
                normalization=normalization,
                down_sampling_layer=down_sampling_layer))

    def forward(self, inputs, label_onehots=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        base_feature = x.view(inputs.size(0), -1)
        x = self.dense(base_feature)
        return x.view(inputs.size(0))


class SNGANGenerator(nn.Module):
    pass


class SNGANDiscriminator(nn.Module):
    pass

