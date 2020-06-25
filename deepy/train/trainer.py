import time
import os
from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F



class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class Trainer(object):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 scheduler=None,
                 init_epoch=0,
                 device='cpu'):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.device = device
        self.epoch = init_epoch

    def train(self, epochs, valloader=None):
        start_time = time.time()
        start_epoch = self.epoch
        costs = []
        print('-----Training Started-----')
        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            loss = self.step()
            elapsed_time = time.time() - start_time
            ave_required_time = elapsed_time / (epoch + 1)
            finish_time = ave_required_time * (epochs - (epoch + 1))
            format_str = 'Epoch: {:03d}/{:03d}'.format(epoch+1, epochs)
            format_str + ' | '
            format_str += 'Loss: {:.4f}'.format(loss)
            format_str + ' | '
            format_str += 'Time: {:02d} hour {:02.2f} min'.format(elapsed_time/60/60, elapsed_time/60)
            format_str + ' | '
            format_str += 'Finish after: {:02d} hour {:02.2f} min'.format(finish_time/60/60, finish_time/60)
            print(format_str)
            costs.append(loss)
        print('Total Training Time: {:02d} hour {:02.2f} min'.format(elapsed_time/60))
        print('-----Training Finished-----')

        return self.net

    def step(self):
        self.net.train()
        loss_meter = AverageMeter()
        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs.size(0))

        if self.scheduler is not None:
            scheduler.step()
        self.epoch += 1

        return loss_meter.average


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

    def eval(self, dataloader, num_classes):
        self.net.eval()
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels)

                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        class_accuracy = class_correct[i] / class_total[i]
        total_accuracy = sum(class_correct) / sum(class_total)
        return total_accuracy, class_accuracy


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

    def step(self):
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

    def step(self):
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


class AutoEncoderTrainer(torchtrain.trainer.RegressorTrainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 device):
        super(AutoEncoderTrainer, self).__init__(net, optimizer, criterion,
                                                 dataloader, device)

    def step(self):
        self.net.train()

        loss_meter = torchtrain.trainer.AverageMeter()

        for inputs, *_ in self.dataloader:
            targets = inputs.clone().detach()
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
        
        loss_meter = torchtrain.trainer.AverageMeter()

        with torch.no_grad():
            for inputs, *_ in dataloader:
                targets = inputs.clone().detach()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss_meter.update(loss.item(), number=inputs.size(0))
            
        return loss_meter.average
