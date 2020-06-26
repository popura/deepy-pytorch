import sys
import os
import os.path
import pickle

import torch
import torch.utils.data as torchdata

from . import transform as mytf


class ClassDataset(torchdata.Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return torchdata.ConcatDataset([self, other])


class RegDataset(torchdata.Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return torchdata.ConcatDataset([self, other])


class PureDatasetFolder(torchdata.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None,
                 pre_load=False, pre_transform=None, pre_target_transform=None, pre_transforms=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = mytf.SeparatedTransform(transform, target_transform)
        self.transforms = transforms

        self.pre_load = pre_load
        has_pre_transforms = pre_transforms is not None
        has_pre_separate_transform = pre_transform is not None or pre_target_transform is not None
        if has_pre_transforms and has_pre_separate_transform:
            raise ValueError("Only pre_transforms or pre_transform/pre_target_transform can "
                             "be passed as argument")
        
        if has_pre_separate_transform:
            pre_transforms = torchdataset.transform.SeparatedTransform(pre_transform, pre_target_transform)
        self.pre_transforms = pre_transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])
    
    def _pre_process(self):
        raise NotImplementedError

    def extra_repr(self):
        return ""


class UnorganizedDatasetFolder(PureDatasetFolder):
    """A generic data loader where the samples are arranged in this way: ::
        root/xxx.ext
        root/xxy.ext
        root/xxz.ext
        root/123.ext
        root/nsdf3.ext
        root/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        samples (list): List of sample pathes
    """

    def __init__(self, root, loader, extensions=None,
                 transform=None, target_transform=None, transforms=None,
                 pre_load=False, pre_transform=None, pre_target_transform=None, pre_transforms=None,
                 is_valid_file=None):
        super(UnorganizedDatasetFolder, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       transforms=transforms,
                                                       pre_load=pre_load,
                                                       pre_transform=pre_transform,
                                                       pre_target_transform=pre_target_transform,
                                                       pre_transforms=pre_transforms)
        samples = make_unorganized_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        if self.pre_load:
            self._pre_process()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            torch.Tensor: sample where target is class_index of the target class.
        """
        if self.pre_load:
            sample = self.preprocessed_samples[index]
        else:
            path = self.samples[index]
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def _pre_process(self):
        preprocessed_samples = []
        for i in range(len(self)):
            sys.stdout.write("\rloaded {0} / {1}".format(i+1, len(self)))
            sys.stdout.flush()
            path = self.samples[i]
            sample = self.loader(path)
            if self.pre_transform is not None:
                sample = self.pre_transform(sample)
            preprocessed_samples.append(sample)

        self.preprocessed_samples = preprocessed_samples
        sys.stdout.write("\n")


class DatasetFolder(PureDatasetFolder):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None,
                 transform=None, target_transform=None, transforms=None,
                 pre_load=False, pre_transform=None, pre_target_transform=None, pre_transforms=None,
                 is_valid_file=None):
        super(DatasetFolder, self).__init__(root,
                                            transform=transform,
                                            target_transform=target_transform,
                                            transforms=transforms,
                                            pre_load=pre_load,
                                            pre_transform=pre_transform,
                                            pre_target_transform=pre_target_transform,
                                            pre_transforms=pre_transforms)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        if self.pre_load:
            self._pre_process()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

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
        sample, target = self.transforms(sample, target)

        return sample, target

    def __len__(self):
        return len(self.samples)
    
    def _pre_process(self):
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


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances

def make_unorganized_dataset(directory, extensions=None, is_valid_file=None):
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
        raise ValueError("Given path is not a directory")

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                instances.append(path)
    return instances



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


class PickleFolder(DatasetFolder):
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

    def __init__(self, root, transform=None, target_transform=None, transforms=None,
                 pre_load=False, pre_transform=None, pre_target_transform=None, pre_transforms=None,
                 loader=pickle_loader, is_valid_file=None):
        super(PickleFolder, self).__init__(root, loader, PICKLE_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          transforms=transforms,
                                          pre_load=pre_load,
                                          pre_transform=pre_transform,
                                          pre_target_transform=pre_target_transform,
                                          pre_transforms=pre_transforms,
                                          is_valid_file=is_valid_file)
        self.pickles = self.samples
