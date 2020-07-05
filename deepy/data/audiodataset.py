import sys
import os
import os.path

import torch
import torch.utils.data as torchdata
import torchaudio

from . import dataset


AUDIO_EXTENSIONS = (".wav", ".mp3")


def default_loader(path):
    """The default loader for audio tracks where sampling rate will be ignored
    Args:
        path: Path to an audio track
    
    Returns:
        waveform: waveform of the audio track
    """
    waveform, sampling_rate = torchaudio.load(path)
    return waveform


class UnorganizedAudioFolder(dataset.UnorganizedDatasetFolder):
    """A audio data loader where the audio tracks are arranged in this way: ::
        root/xxx.wav
        root/xxy.wav
        root/xxz.wav
        root/123.wav
        root/nsdf3.wav
        root/asd932_.wav
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
    def __init__(self, root, transform=None, pre_load=False, pre_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(UnorganizedAudioFolder, self).__init__(root, loader, AUDIO_EXTENSIONS if is_valid_file is None else None,
                                                     transform=transform,
                                                     pre_load=pre_load, pre_transform=pre_transform,
                                                     is_valid_file=is_valid_file)
        self.tracks = self.samples


class AudioFolder(dataset.DatasetFolder):
    """A generic data loader where the audio tracks are arranged in this way: ::

        root/car/xxx.wav
        root/car/xxy.wav
        root/car/xxz.wav

        root/home/123.wav
        root/home/nsdf3.wav
        root/home/asd932_.wav

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
        tracks (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, transforms=None,
                 pre_load=False, pre_tranform=None, pre_target_transform=None, pre_transforms=None,
                 loader=default_loader, is_valid_file=None):
        super(AudioFolder, self).__init__(root, loader, AUDIO_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          transforms=transforms,
                                          pre_load=pre_load,
                                          pre_transform=pre_transform,
                                          pre_target_transform=pre_target_transform,
                                          pre_transforms=pre_transforms,
                                          is_valid_file=is_valid_file)
        self.tracks = self.samples
    

class PreLoadAudioFolder(AudioFolder):
    """A generic data loader storing all transformed data on memory,
        where the audio tracks are arranged in this way: ::

        root/car/xxx.wav
        root/car/xxy.wav
        root/car/xxz.wav

        root/home/123.wav
        root/home/nsdf3.wav
        root/home/asd932_.wav

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

    def __init__(self, *args, **kwargs):
        super(PreLoadAudioFolder, self).__init__(*args, **kwargs)
        self.load_all()
        raise FutureWarning
    
    def load_all(self):
        preprocessed_samples = []
        for i in range(len(self)):
            sys.stdout.write("\rloaded {0} / {1}".format(i+1, len(self)))
            sys.stdout.flush()
            path, target = self.samples[i]
            sample = self.loader(path)

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            preprocessed_samples.append((sample, target))

        self.preprocessed_samples = preprocessed_samples
        sys.stdout.write("\n")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return self.preprocessed_samples[index]