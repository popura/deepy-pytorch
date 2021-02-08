import random

import torch

from deepy.data.transform import Transform, SeparatedTransform
from deepy.data.transform import PairedTransform, PairedCompose, ToPairedTransform
from deepy.nn import functional as myF


class RandomCrop(Transform):
    def __init__(self, length: int, generator=None):
        self.length = length
        self.generator = generator
    
    def __call__(self, data):
        signal_length = data.size(-1)
        start_index = torch.randint(0, signal_length - self.length + 1,
                                    (1, ),
                                    generator=self.generator)
        end_index = start_index + self.length
        return data[..., start_index:end_index]
    
    def __repr__(self):
        return self.__class__.__name__ + '(length={})'.format(self.length)


class RandomFrames(RandomCrop):
    def __init__(self, n_frames=5, generator=None):
        super().__init__(length=n_frames, generator=generator)
        self.n_frames = n_frames

    def __repr__(self):
        return self.__class__.__name__ + '(n_frames={})'.format(self.n_frames)


class Windowing(Transform):
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


class Plane2Vector(Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.cat([data[..., i, :] for i in range(data.size(-2))], dim=-1)
        


class ToVector(Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        return data.reshape(-1, )
    
    def __repr__(self):
        return self.__class__.__name__


class PickUpChannel(Transform):
    def __init__(self, chidx=0):
        self.chidx = chidx

    def __call__(self, data):
        return data[self.chidx]
    
    def __repr__(self):
        return self.__class__.__name__ + '(chidx={})'.format(self.chidx)
