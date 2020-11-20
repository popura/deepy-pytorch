import time
import os
import typing

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from deepy.train.trainer import AverageMeter, Trainer


class RegressorTrainer(Trainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 scheduler=None,
                 extensions=None,
                 init_epoch=0,
                 device='cpu'):
        super(RegressorTrainer, self).__init__(
            net, optimizer, criterion, dataloader,
            scheduler=scheduler, extensions=extensions,
            init_epoch=init_epoch,
            device=device)

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
            
        ave_loss = loss_meter.average
        return {'loss': ave_loss}
