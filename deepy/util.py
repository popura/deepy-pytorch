import numpy as np
import matplotlib.pyplot as plt
import functools


def gradshow(grad):
    npgrad = grad.numpy()
    npgrad = (npgrad - np.min(npgrad)) / (np.max(npgrad) - np.min(npgrad))
    print(npgrad)
    plt.imshow(npgrad[0, 0, :, :], cmap="gray")
    plt.show()
