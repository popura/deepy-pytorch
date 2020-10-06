import sys
import os
import os.path
import pickle

import torch
import torch.utils.data as torchdata

from . import transform as mytf
from .dataset import ClassDataset, RegDataset


class ToyClassDataset(ClassDataset):
    def __init__(self, num_classes: int):
        super(ToyClassDataset, self).__init__()
        self.num_classes = num_classes
    
    def __getitem__(self, index):
        sample = index
        target = index % self.num_classes
        return sample, target
    
    def __len__(self):
        return 100


class ToyRegDataset(RegDataset):
    def __init__(self):
        super(ToyRegDataset, self).__init__()
    
    def __getitem__(self, index):
        sample = index
        target = index**2
        return sample, target
    
    def __len__(self):
        return 100