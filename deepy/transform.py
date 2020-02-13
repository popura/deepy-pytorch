from abc import ABCMeta
from abc import abstractmethod
import random
import torchvision.functional as F
from torchvision.transforms import RandomResizedCrop
from PIL import Image


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class PairedTransform(metaclass = ABCMeta):
    @abstractmethod
    def __call__(self, img, target):
        pass


class ToPairedTransform(PairedTransform):
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img, target):
        return self.transform(img), self.transform(target)

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.transform)


class Compose(PairedTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class PairedRandomHorizontalFlip(PairedTransform):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, target):
        if random.random() > self.p:
            return F.hflip(img), F.hflip(target)
        return img, target
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class PairedRandomResizedCrop(PairedTransform):
    def __init__(self, size, scale=(0.08, 1.0),
                 ratio=(3./4., 4./3.), interpolation=Image.BILINEAR):
        self.transform = RandomResizedCrop(size, scale,
                                           ratio, interpolation)
    
    def __call__(self, img, target):
        i, j, h, w = self.transform.get_params(img, self.transform.scale
                                               self.transform.ratio)
        transformed_img = F.resized_crop(img, i, j, h, w, self.transform.size,
                                         self.transform.interpolation)
        transformed_target = F.resized_crop(target, i, j, h, w, self.transform.size,
                                            self.transform.interpolation)
        return transformed_img, transformed_target
    
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
