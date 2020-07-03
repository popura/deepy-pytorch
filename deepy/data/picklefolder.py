import sys
import os
import os.path
import pickle

from .dataset import DatasetFolder



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