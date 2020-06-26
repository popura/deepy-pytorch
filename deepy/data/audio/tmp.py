import sys
import os
import os.path
import random
import pickle
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
import torchdataset
from torchdataset.audiodataset import AUDIO_EXTENSIONS, default_loader
from torchdataset.dataset import has_file_allowed_extension


class RandomFrames(torchdataset.transform.Transform):
    def __init__(self, n_frames=5):
        self.n_frames = n_frames

    def __call__(self, data):
        total_frames = data.size(-1)
        start_frame = random.randint(0, total_frames-self.n_frames)
        end_frame = start_frame + self.n_frames
        return data[..., start_frame:end_frame]

    def __repr__(self):
        return self.__class__.__name__ + '(n_frames={})'.format(self.n_frames)


class Windowing(torchdataset.transform.Transform):
    def __init__(self, n_frames=5, stride=1, n_signals=None):
        self.n_frames = n_frames
        if not stride == 1:
            raise NotImplementedError
        self.stride = stride
        self.n_signals = n_signals
    
    def __call__(self, data):
        total_frames = data.size(-1)

        if self.n_signals == None:
            n_signals = total_frames - self.n_frames + 1
        else:
            n_signals = self.n_signals

        return torch.stack([data[..., i:i+self.n_frames] for i in range(n_signals)], dim=1)

    def __repr__(self):
        return self.__class__.__name__ + '(n_frames={}, stride={})'.format(self.n_frames, self.stride)


class Plane2Vector(torchdataset.transform.Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.cat([data[..., i, :] for i in range(data.size(-2))], dim=-1)
        


class ToVector(torchdataset.transform.Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        return data.reshape(-1, )
    
    def __repr__(self):
        return self.__class__.__name__


class PickUpChannel(torchdataset.transform.Transform):
    def __init__(self, chidx=0):
        self.chidx = chidx

    def __call__(self, data):
        return data[self.chidx]
    
    def __repr__(self):
        return self.__class__.__name__ + '(chidx={})'.format(self.chidx)


PICKLE_EXTENSIONS = (".pkl")


def pickle_loader(path):
    """A loader for pickle files that contain a sample
    Args:
        path: Path to an audio track
    
    Returns:
        sample: A sample
    """
    with open(path, 'rb') as pkl:
        sample = pickle.load(pkl)
    return sample


class PickleFolder(torchdataset.dataset.DatasetFolder):
    """A generic data loader where the pickle files are arranged in this way: ::
        root/car/xxx.pkl
        root/car/xxy.pkl
        root/car/xxz.pkl
        root/home/123.pkl
        root/home/nsdf3.pkl
        root/home/asd932_.pkl
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=pickle_loader, is_valid_file=None):
        super(PickleFolder, self).__init__(root, loader, PICKLE_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.pickles = self.samples


MACHINE_NAMES = ('ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve')


class DCASE2019Task1Dataset(torchdataset.dataset.PureDatasetFolder):
    """
    """
    def __init__(self, root, mode, loader=default_loader, extensions=AUDIO_EXTENSIONS,
                 transforms=None, transform=None, target_transform=None,
                 is_valid_file=None,
                 pre_load=False, pre_transform=None,
                 pre_target_transform=None, pre_transforms=None):
        super(DCASE2019Task1Dataset, self).__init__(root,
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
