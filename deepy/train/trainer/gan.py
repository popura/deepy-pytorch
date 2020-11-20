import time
import os
import typing

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from deepy.train.trainer import AverageMeter, ABCTrainer


class GANTrainer(ABCTrainer):
    def __init__(self,
                 net_g,
                 net_d,
                 optimizer_g,
                 optimizer_d,
                 dataloader,
                 scheduler_g=None,
                 scheduler_d=None,
                 extensions=None,
                 init_epoch=0,
                 latent_dim=100,
                 dsc_train_ratio=1,
                 device='cpu'):
        super().__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.dataloader = dataloader
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.extensions = extensions
        self.device = device
        self.latent_dim = latent_dim
        self.dsc_train_ratio = dsc_train_ratio
        self.epoch = init_epoch
        self.history = {}

    def train(self, epochs, *args, **kwargs):
        start_time = time.time()
        start_epoch = self.epoch
        self.history["train"] = []
        # self.history["validation"] = []
        print('-----Training Started-----')
        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            # losses is a dictionary {str: value} and self.epoch is incremented in this function
            # (i.e. self.epoch = epoch + 1)
            losses = self.step()
            # vallosses is a dictionary {str: value}
            # vallosses = self.eval(*args, **kwargs)
            elapsed_time = time.time() - start_time

            self.history["train"].append({'epoch':self.epoch, 'loss':loss})
            # self.history["validation"].append({'epoch':self.epoch, **vallosses})

            self.extend()

            ave_required_time = elapsed_time / self.epoch
            finish_time = ave_required_time * (epochs - self.epoch)
            format_str = 'epoch: {:03d}/{:03d}'.format(self.epoch, epochs)
            format_str += ' | '
            format_str += 'loss: {:.4f}'.format(loss)
            format_str += ' | '
            # if vallosses is not None:
            #     for k, v in vallosses.items():
            #         format_str += '{}: {:.4f}'.format(k, v)
            #         format_str += ' | '
            format_str += 'time: {:02d} hour {:02.2f} min'.format(int(elapsed_time/60/60), elapsed_time/60%60)
            format_str += ' | '
            format_str += 'finish after: {:02d} hour {:02.2f} min'.format(int(finish_time/60/60), finish_time/60%60)
            print(format_str)
        print('Total training time: {:02d} hour {:02.2f} min'.format(int(elapsed_time/60/60), elapsed_time/60%60))
        print('-----Training Finished-----')

        return self.net_g, self.net_d

    def step(self):
        self.net_g.train()
        self.net_d.train()

        loss_g_meter = AverageMeter()
        loss_d_meter = AverageMeter()

        for i, (real, _) in enumerate(self.dataloader):
            real = real.to(self.device)
            # train generator
            if i % self.dsc_train_ratio == 0:
                loss_g = self.step_g(real)
                loss_g_meter.update(loss_g.item(), number=real.size(0))

            # train discriminator
            loss_d = self.step_d(real)
            loss_d_meter.update(loss_d.item(), number=real.size(0))

        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()
        self.epoch += 1
        ave_loss_g = loss_g_meter.average
        ave_loss_d = loss_d_meter.average

        return {'loss_g': ave_loss_g, 'loss_d': ave_loss_d}
    
    def step_g(self, real_data):
        z = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
        fake = self.net_g(z)

        discr_pred = self.net_d(fake)
        loss_g = self._criterion_g(discr_pred)

        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()
        return loss_g
    
    def step_d(self, real_data):
        z = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
        fake_data = self.net_g(z).detach()

        discr_pred_real = self.net_d(real_data)
        discr_pred_fake = self.net_d(fake_data)
        loss_d = self._criterion_d(discr_pred_real, discr_pred_fake)

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d
    
    def _criterion_g(self, predicted_label):
        valid_label = torch.ones(predicted_label.size(0), dtype=torch.float).to(self.device)
        loss_g = F.binary_cross_entropy_with_logits(predicted_label, valid_label)
        return loss_g
    
    def _criterion_d(self, predicted_label_real, predicted_label_fake):
        valid_label = torch.ones(predicted_label_real.size(0), dtype=torch.float).to(self.device)
        fake_label = torch.ones(predicted_label_fake.size(0), dtype=torch.float).to(self.device)
        real_loss = F.binary_cross_entropy_with_logits(predicted_label_real,
                                                       valid_label)
        fake_loss = F.binary_cross_entropy_with_logits(predicted_label_fake,
                                                       fake_label)
        loss_d = 0.5 * (real_loss + fake_loss)
        return loss_d
    
    def eval(self):
        raise NotImplementedError()

    def extend(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            if extension.trigger(self):
                extension(self)
        return

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
                 scheduler_g=None,
                 scheduler_d=None,
                 extensions=None,
                 init_epoch=0,
                 latent_dim=100,
                 dsc_train_ratio=1,
                 c=10.0,
                 device='cpu'):
        super().__init__(net_g=net_g,
                         net_d=net_d,
                         optimizer_g=optimizer_g,
                         optimizer_d=optimizer_d,
                         dataloader=dataloader,
                         scheduler_g=None,
                         scheduler_d=None,
                         extensions=None,
                         init_epoch=0,
                         latent_dim=100,
                         dsc_train_ratio=1,
                         device='cpu')
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