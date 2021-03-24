import torch
import torch.nn as nn
import torch.nn.functional as F
from deepy import layer


class ConvVAEEncoder(nn.Module):
    """Encoder of Convolutional VAE

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ConvVAEEncoder(10, device).to(device)
    >>> summary(net , (3, 64, 64))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]             448
                Conv2d-2           [-1, 32, 16, 16]           4,640
                Conv2d-3             [-1, 64, 8, 8]          18,496
                Conv2d-4             [-1, 64, 4, 4]          36,928
                Conv2d-5             [-1, 64, 2, 2]          36,928
                Linear-6                   [-1, 10]           2,570
                Linear-7                   [-1, 10]           2,570
    ================================================================
    Total params: 102,580
    Trainable params: 102,580
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.23
    Params size (MB): 0.39
    Estimated Total Size (MB): 0.67
    ----------------------------------------------------------------
    """
    def __init__(self, latent_dim, device, down_conv_layer=nn.Conv2d):
        super(ConvVAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.conv_1 = down_conv_layer(in_channels=3,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_2 = down_conv_layer(in_channels=16,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_3 = down_conv_layer(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_4 = down_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.conv_5 = down_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        self.z_mean = nn.Linear(64*2*2, self.latent_dim)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = nn.Linear(64*2*2, self.latent_dim)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, features):
        x = self.conv_1(features)
        x = F.leaky_relu(x)
        #print('conv1 out:', x.size())

        x = self.conv_2(x)
        x = F.leaky_relu(x)
        #print('conv2 out:', x.size())

        x = self.conv_3(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        x = self.conv_4(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        x = self.conv_5(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        z_mean = self.z_mean(x.view(-1, 64*2*2))
        z_log_var = self.z_log_var(x.view(-1, 64*2*2))
        encoded = self.reparameterize(z_mean, z_log_var)

        return z_mean, z_log_var, encoded


class ConvVAEDecoder(nn.Module):
    """Decoder of Convolutional VAE

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ConvVAEDecoder(10).to(device)
    >>> summary(net , (10, ))
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                  [-1, 256]           2,816
       ConvTranspose2d-2             [-1, 64, 4, 4]          65,600
       ConvTranspose2d-3             [-1, 64, 8, 8]          65,600
       ConvTranspose2d-4           [-1, 32, 16, 16]          32,800
       ConvTranspose2d-5           [-1, 16, 32, 32]           8,208
       ConvTranspose2d-6            [-1, 3, 64, 64]             771
    ================================================================
    Total params: 175,795
    Trainable params: 175,795
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.32
    Params size (MB): 0.67
    Estimated Total Size (MB): 0.99
    ----------------------------------------------------------------
    """
    def __init__(self, latent_dim, up_conv_layer=nn.ConvTranspose2d):
        super(ConvVAEDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.linear_1 = nn.Linear(self.latent_dim, 64*2*2)

        self.deconv_1 = up_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_2 = up_conv_layer(in_channels=64,
                                      out_channels=64,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_3 = up_conv_layer(in_channels=64,
                                      out_channels=32,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_4 = up_conv_layer(in_channels=32,
                                      out_channels=16,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

        self.deconv_5 = up_conv_layer(in_channels=16,
                                      out_channels=3,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1)

    def forward(self, encoded):
        x = self.linear_1(encoded)
        x = x.view(-1, 64, 2, 2)

        x = self.deconv_1(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.deconv_2(x)
        x = F.leaky_relu(x)
        #print('deconv2 out:', x.size())

        x = self.deconv_3(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.deconv_4(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.deconv_5(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        decoded = torch.sigmoid(x)
        return decoded
