class Transform(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transform.Compose([
        >>>     transform.CenterCrop(10),
        >>>     transform.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class SeparatedTransform(Transform):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, data, target):
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


class PairedTransform(Transform):
    def __call__(self, data, target):
        raise NotImplementedError()


class PairedCompose(PairedTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, target):
        for t in self.transforms:
            data, target = t(data, target)
        return data, target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToPairedTransform(PairedTransform):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data, target):
        return self.transform(data), self.transform(target)

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.transform)


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, data):
        return self.lambd(data)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Pad(Transform):
    def __init__(self, pad, mode='constant', value=0):
        self.pad = pad
        self.mode = mode
        self.value = value
    
    def __call__(self, data):
        return torch.nn.functional.pad(data, self.pad, self.mode, self.value)
    
    def __repr__(self):
        return self.__class__.__name__ + '(pad={}, mode={}, value={})'.format(self.pad, self.mode, self.value)
