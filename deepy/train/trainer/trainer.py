import time
import os
from abc import ABC
from abc import abstractmethod
import typing

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


class ABCTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    @abstractmethod
    def eval(self):
        raise NotImplementedError()

    @abstractmethod
    def extend(self):
        raise NotImplementedError()


class Trainer(ABCTrainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 scheduler=None,
                 extensions=None,
                 init_epoch=0,
                 device='cpu'):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.extensions = extensions
        self.device = device
        self.epoch = init_epoch
        self.history = {}

    def train(self, epochs, *args, **kwargs):
        start_time = time.time()
        start_epoch = self.epoch
        self.history["train"] = []
        self.history["validation"] = []
        print('-----Training Started-----')
        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            # loss is a scalar and self.epoch is incremented in this function
            # (i.e. self.epoch = epoch + 1)
            loss = self.step()
            # vallosses is a dictionary {str: value}
            vallosses = self.eval(*args, **kwargs)
            elapsed_time = time.time() - start_time

            self.history["train"].append({'epoch':self.epoch, 'loss':loss})
            self.history["validation"].append({'epoch':self.epoch, **vallosses})

            self.extend()

            ave_required_time = elapsed_time / self.epoch
            finish_time = ave_required_time * (epochs - self.epoch)
            format_str = 'epoch: {:03d}/{:03d}'.format(self.epoch, epochs)
            format_str += ' | '
            format_str += 'loss: {:.4f}'.format(loss)
            format_str += ' | '
            if vallosses is not None:
                for k, v in vallosses.items():
                    format_str += '{}: {:.4f}'.format('Val. ' + k, v)
                    format_str += ' | '
            format_str += 'time: {:02d} hour {:02.2f} min'.format(int(elapsed_time/60/60), elapsed_time/60%60)
            format_str += ' | '
            format_str += 'finish after: {:02d} hour {:02.2f} min'.format(int(finish_time/60/60), finish_time/60%60)
            print(format_str)
        print('Total training time: {:02d} hour {:02.2f} min'.format(int(elapsed_time/60/60), elapsed_time/60%60))
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
            self.scheduler.step()
        self.epoch += 1
        ave_loss = loss_meter.average

        return ave_loss

    def eval(self, dataloader=None):
        raise NotImplementedError()

    def extend(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            if extension.trigger(self):
                extension(self)
        return