import os
from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from torchvision.utils import save_image

from deepy.util import AverageMeter, gradshow


class Trainer(metaclass = ABCMeta):
    @abstractmethod
    def train(self):
        pass


class ClassifierTrainer(Trainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 device):
        super(ClassifierTrainer, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def train(self):
        self.net.train()

        loss_meter = AverageMeter()

        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            #if i % 500 == 500-1:
            #    inputs.requires_grad_(requires_grad=True)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs.size(0))

        return loss_meter.average

    def eval(self, dataloader):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def visualize_grad(self, dataloader):
        self.net.train()
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        features = list(self.net.net)
        features = nn.ModuleList(features).eval()
        results = []
        x = inputs
        for ii, model in enumerate(features):
            x = model(x)
            x.retain_grad()
            results.append(x)

        loss = self.criterion(x.view(inputs.size(0), -1), labels)
        # loss.backward()
        results[3].backward(torch.ones_like(results[3]))

        for x in results:
            if x.grad is not None:
                gradshow(x.grad.cpu().detach())


class RegressorTrainer(Trainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 device):
        super(RegressorTrainer, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def train(self):
        self.net.train()

        loss_meter = AverageMeter()

        for inputs, targets in self.dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs.size(0))

        return loss_meter.average

    def eval(self, dataloader):
        self.net.eval()
        
        loss_meter = AverageMeter()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss_meter.update(loss.item(), number=inputs.size(0))
            
        return loss_meter.average


class GANTrainer(Trainer):
    def __init__(self,
                 net_g,
                 net_d,
                 optimizer_g,
                 optimizer_d,
                 dataloader,
                 device,
                 latent_dim=100,
                 dsc_train_ratio=1):
        super(GANTrainer, self).__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.dataloader = dataloader
        self.device = device
        self.latent_dim = latent_dim
        self.dsc_train_ratio = dsc_train_ratio

    def train(self):
        self.net_g.train()
        self.net_d.train()

        loss_g_meter = AverageMeter()
        loss_d_meter = AverageMeter()

        for i, (real, _) in enumerate(self.dataloader):
            valid_label = torch.ones(real.size(0)).float().to(self.device)
            fake_label = torch.zeros(real.size(0)).float().to(self.device)

            # train generator
            if i % self.dsc_train_ratio == 0:
                z = torch.randn(real.size(0), self.latent_dim).to(self.device)
                fake = self.net_g(z)

                discr_pred = self.net_d(fake)
                loss_g = F.binary_cross_entropy_with_logits(discr_pred,
                                                            valid_label)

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                loss_g_meter.update(loss_g.item(), number=real.size(0))

            # train discriminator
            z = torch.randn(real.size(0), self.latent_dim).to(self.device)
            real = real.to(self.device)
            fake = self.net_g(z).detach()

            discr_pred_real = self.net_d(real)
            discr_pred_fake = self.net_d(fake)
            real_loss = F.binary_cross_entropy_with_logits(discr_pred_real,
                                                           valid_label)
            fake_loss = F.binary_cross_entropy_with_logits(discr_pred_fake,
                                                           fake_label)
            loss_d = 0.5 * (real_loss + fake_loss)

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

            loss_d_meter.update(loss_d.item(), number=real.size(0))

        return loss_g_meter.average, loss_d_meter.average

    def save_sample(self, filename):
        self.net_g.eval()

        z = torch.randn(64, self.latent_dim).to(self.device)

        with torch.no_grad():
            fake = self.net_g(z)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(fake.data, filename, normalize=True)


class WGANgpTrainer(GANTrainer):
    def __init__(self,
                 net_g,
                 net_d,
                 optimizer_g,
                 optimizer_d,
                 dataloader,
                 device,
                 latent_dim=100,
                 dsc_train_ratio=5,
                 c=10.0):
        super(WGANgpTrainer, self).__init__(net_g, net_d, optimizer_g,
                                            optimizer_d, dataloader, device,
                                            latent_dim, dsc_train_ratio)
        self.c = c

    def train(self):
        self.net_g.train()

        loss_g_meter = AverageMeter()
        loss_d_meter = AverageMeter()

        for i, (real, _) in enumerate(self.dataloader):
            # train generator
            if i % self.dsc_train_ratio == 0:
                z = torch.randn(real.size(0), self.latent_dim).to(self.device)

                fake = self.net_g(z)
                loss_g = -self.net_d(fake).mean()

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                loss_g_meter.update(loss_g.item(), number=real.size(0))

            # train discriminator
            real = real.to(self.device)
            z = torch.randn(real.size(0), self.latent_dim).to(self.device)

            fake = self.net_g(z).detach()

            loss_d = -self.net_d(real).mean() + self.net_d(fake).mean(
            ) + self.c * self.gradient_penalty(real, fake).mean()

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

            loss_d_meter.update(loss_d.item(), number=real.size(0))

        return loss_g_meter.average, loss_d_meter.average

    def gradient_penalty(self, real, fake):
        epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)

        interpolates = (epsilon * real + (1 - epsilon) * fake).clone().detach()
        interpolates.requires_grad_(True)
        gradients = autograd.grad(
            self.net_d(interpolates),
            interpolates,
            grad_outputs=torch.ones(real.size(0)).to(self.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        return (gradients.contiguous().view(real.size(0), -1).norm(2, dim=1) - 1).pow(2)


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
