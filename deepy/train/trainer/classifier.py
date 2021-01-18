import time
import os
import typing

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from deepy.train.trainer import AverageMeter, Trainer


class ClassifierTrainer(Trainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 scheduler=None,
                 extensions=None,
                 init_epoch=0,
                 device='cpu'):
        super().__init__(
            net, optimizer, criterion, dataloader,
            scheduler=scheduler, extensions=extensions,
            init_epoch=init_epoch,
            device=device)

    def eval(self, dataloader, classes):
        self.net.eval()
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
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

        class_accuracy = [c / t for c, t in zip(class_correct, class_total)]
        total_accuracy = sum(class_correct) / sum(class_total)

        hist_dict = {'total acc': total_accuracy}
        hist_dict.update({classes[i]: class_accuracy[i] for i in range(len(classes))})
        return hist_dict


class MultiInputClassifierTrainer(Trainer):
    """

    Args:
        net: an nn.Module. It should be applicable for multiple inputs
        dataloaders: len(dataloaders) should be equal to the number of inputs
                     and len(dataloader) for each dataloader should be the same as each other.
    """
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloaders,
                 scheduler=None,
                 extensions=None,
                 init_epoch=0,
                 device='cpu'):
        super().__init__(
            net, optimizer, criterion, dataloader=None,
            scheduler=scheduler, extensions=extensions,
            init_epoch=init_epoch, device=device)
        self.dataloaders = dataloaders

    def step(self):
        self.net.train()
        loss_meter = AverageMeter()
        dl_iters = [self.dataloaders[i].__iter__()
                    for i in range(len(self.dataloaders))]
        for i in range(len(self.dataloaders[0])):
            inputs = []
            for j in range(len(self.dataloaders)):
                tmp_inputs, tmp_labels = dl_iters[j].next()
                tmp_inputs = tmp_inputs.to(self.device)
                tmp_labels = tmp_labels.to(self.device)
                if j == 0:
                    labels = tmp_labels
                else:
                    if not (labels == tmp_labels).all():
                        raise ValueError('Different labels are loaded')
                inputs.append(tmp_inputs)

            outputs = self.net(*inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs[0].size(0))

        if self.scheduler is not None:
            self.scheduler.step()
        self.epoch += 1
        ave_loss= loss_meter.average

        return ave_loss

    def eval(self, dataloaders, classes):
        self.net.eval()
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            dl_iters = [dataloaders[i].__iter__()
                        for i in range(len(dataloaders))]
            for i in range(len(dataloaders[0])):
                inputs = []
                for j in range(len(dataloaders)):
                    tmp_inputs, tmp_labels = dl_iters[j].next()
                    tmp_inputs = tmp_inputs.to(self.device)
                    tmp_labels = tmp_labels.to(self.device)
                    if j == 0:
                        labels = tmp_labels
                    else:
                        if not (labels == tmp_labels).all():
                            raise ValueError('Different labels are loaded')
                    inputs.append(tmp_inputs)

                outputs = self.net(*inputs)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels)

                for j in range(len(labels)):
                    label = labels[j]
                    class_correct[label] += c[j].item()
                    class_total[label] += 1

        class_accuracy = [c / t for c, t in zip(class_correct, class_total)]
        total_accuracy = sum(class_correct) / sum(class_total)

        hist_dict = {'total acc': total_accuracy}
        hist_dict.update({classes[i]: class_accuracy[i] for i in range(len(classes))})
        return hist_dict
    