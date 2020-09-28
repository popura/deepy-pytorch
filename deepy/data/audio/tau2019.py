import sys
import os
import os.path
import random
from pathlib import Path

import torch
import torchaudio
from .audiodataset import AUDIO_EXTENSIONS, default_loader
from ..dataset import PureDatasetFolder, has_file_allowed_extension


class TAU2019(PureDatasetFolder):
    """TAU urban acoustic scene 2019 dataset.
       This dataset was used for DCASE 2019 Task 1.
       For using this dataset, download the dataset from the following links:
       https://zenodo.org/record/2589280#.XvWs0Zbgprk
       https://zenodo.org/record/3063822#.XvWs55bgprk
       Then, unzip them in the *root* folder.
    """
    def __init__(self, root, mode, loader=default_loader, extensions=AUDIO_EXTENSIONS,
                 transforms=None, transform=None, target_transform=None,
                 is_valid_file=None,
                 pre_load=False, pre_transform=None,
                 pre_target_transform=None, pre_transforms=None):
        super(TAU2019, self).__init__(root,
                                      transforms=transforms,
                                      transform=transform,
                                      target_transform=target_transform)
        self.MODES = ('train', 'evaluate', 'test')

        if mode not in self.MODES:
            raise ValueError("mode \"{}\" is not in {}".format(mode, self.MODES))
        self.mode = mode

        classes, class_to_idx = self._define_classes()
        samples = self._make_dataset(str(self.root), mode,
                                     class_to_idx, extensions, is_valid_file)
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx

        has_pre_transforms = pre_transforms is not None
        has_pre_separate_transform = pre_transform is not None or pre_target_transform is not None
        if has_pre_transforms and has_pre_separate_transform:
            raise ValueError("Only pre_transforms or pre_transform/pre_target_transform can "
                             "be passed as argument")
        if has_pre_separate_transform:
            pre_transforms = torchdataset.transform.SeparatedTransform(pre_transform, pre_target_transform)
        self.pre_transforms = pre_transforms
        self.pre_load = pre_load
        if pre_load:
            self.pre_process()

    def pre_process(self, ):
        preprocessed_samples = []
        for i in range(len(self)):
            sys.stdout.write("\rloaded {0} / {1}".format(i+1, len(self)))
            sys.stdout.flush()
            path, target = self.samples[i]
            sample = self.loader(path)
            if self.pre_transforms is not None:
                sample, target = self.pre_transforms(sample, target)
            preprocessed_samples.append((sample, target))

        self.preprocessed_samples = preprocessed_samples
        sys.stdout.write("\n")
    
    def _define_classes(self, ):
        classes = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
                   'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def _make_dataset(self, directory, mode, class_to_idx, extensions=None, is_valid_file=None):
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
        with open(os.path.join(directory, 'evaluation_setup', 'fold1_'+mode+'.csv')) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.rstrip('\n')
                fname = line.split('\t')[0]
                path = os.path.join(directory, fname)
                class_index = class_to_idx[os.path.split(fname)[1].split('-')[0]]
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
        if self.pre_load:
            sample, target = self.preprocessed_samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)
