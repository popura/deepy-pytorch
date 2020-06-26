import torch
import torch.nn as nn
import torch.nn.functional as F


def softstaircase(input, levels, temperature):
    '''
    (Tensor, Tensor, float) -> Tensor
    '''
    n_steps = len(levels)
    stacked = torch.stack([input for i in range(n_steps)], dim=-1)
    if temperature == 0.:
        return levels[torch.argmin(torch.square(stacked - levels), dim=-1)]
    
    tmp_term = torch.exp(-1. * torch.square(stacked - levels) / (2. * temperature))
    return torch.sum(levels * tmp_term, dim=-1) / torch.sum(tmp_term, dim=-1)
    

class SoftStaircase(nn.Module):
    def __init__(self, levels, temperature):
        super(SoftStaircase, self).__init__()
        self.levels = levels
        self.temperature = temperature
    
    def forward(self, input):
        return softstaircase(input, self.levels, self.temperature)



class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        channel = img1.size(1)

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class DSSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(DSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        channel = img1.size(1)

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel


        return (1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average))

class Sin(torch.nn.Module):
    def __init__(self, ):
        super(Sin, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(tensor)