import time
import os
import typing

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from deepy.train.trainer import AverageMeter, ABCTrainer, Trainer


class VAETrainer(Trainer):
    def __init__(self,
                 encoder,
                 decoder,
                 optimizer,
                 dataloader,
                 device,
                 latent_dim=100):
        super(VAETrainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.latent_dim = latent_dim

    def save_sample(self, filename):
        self.decoder.eval()

        z = torch.randn(64, self.latent_dim).to(self.device)

        with torch.no_grad():
            fake = self.decoder(z)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(fake.data, filename, normalize=True)

    def train(self):
        self.encoder.train()
        self.decoder.train()

        loss_meter = AverageMeter()

        for real, _ in self.dataloader:
            real = real.to(self.device)

            z_mean, z_log_var, encoded = self.encoder(real)
            decoded = self.decoder(encoded)

            # loss = reconstruction loss + Kullback-Leibler divergence
            kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var)
                                    - z_log_var - 1)).sum()
            pixelwise_bce = F.binary_cross_entropy(decoded, real,
                                                   reduction='sum')
            loss = kl_divergence + pixelwise_bce

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=real.size(0))

        return loss_meter.average
