import sys
import os
import os.path
import random
from pathlib import Path

import torch
import torchaudio
from .audiodataset import AUDIO_EXTENSIONS, default_loader
from ..dataset import PureDatasetFolder, has_file_allowed_extension

MACHINE_NAMES = ('ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve')

class DCASE2020Task2(PureDatasetFolder):
    """ A dataset for acoustic anomaly detection.
        This dataset was used for DCASE 2020 task 2.
    """
    def __init__(self, root, machine_name, mode, loader, extensions=None,
                 transforms=None, transform=None, target_transform=None,
                 is_valid_file=None):
        super(DCASE2020Task2Dataset, self).__init__(root,
                                                    transforms=transforms,
                                                    transform=transform,
                                                    target_transform=target_transform)
        self.MACHINE_NAMES = MACHINE_NAMES
        self.MODES = ('train', 'test')

        if machine_name not in self.MACHINE_NAMES:
            raise ValueError("machine_name \"{}\" does not exist".format(machine_name))
        self.machine_name = machine_name
        
        if mode not in self.MODES:
            raise ValueError("mode \"{}\" is not in {}".format(mode, self.MODES))
        self.mode = mode

        classes, class_to_idx = self._define_classes()
        samples = self._make_dataset(str(Path(self.root) / machine_name / mode),
                                     class_to_idx, extensions, is_valid_file)
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx
    
    def _define_classes(self, ):
        classes = ('normal', 'anomaly')
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def _make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    if 'anomaly' in fname:
                        class_index = class_to_idx['anomaly']
                    elif 'normal' in fname:
                        class_index = class_to_idx['normal']
                    else:
                        class_index = -1
                    item = path, class_index
                    instances.append(item)
        return instances
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)
