import numpy as np
import matplotlib.pyplot as plt
import functools


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def gradshow(grad):
    npgrad = grad.numpy()
    npgrad = (npgrad - np.min(npgrad)) / (np.max(npgrad) - np.min(npgrad))
    print(npgrad)
    plt.imshow(npgrad[0, 0, :, :], cmap="gray")
    plt.show()


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


def calc_mapping_function(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    cdf = functools.partial(np.interp, xp=src_values, fp=src_quantiles)
    mapping_func = functools.partial(np.interp, xp=tmpl_quantiles, fp=tmpl_values)
    return cdf, mapping_func


def match_cdf(source, cdf, mapping_func):
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    src_quantiles = cdf(src_values)
    interp_a_values = mapping_func(src_quantiles)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def histoz(data, cdfs=[], mapping_funcs=[]):
    data_r = data.shape[0]
    data_c = data.shape[1] * data.shape[2] * data.shape[3]
    data_tmp = np.reshape(data, (data_r, data_c))
    reference_tmp = np.random.normal(0, 1, (data_r, 1))
    data_tmp2 = np.zeros((data_r, data_c))

    for i in range(data_c):
        data_sep = data_tmp[:, i]
        data_sep = np.expand_dims(data_sep, axis=1)

        # ------------------------------------------------------
        if cdfs == [] or mapping_funcs == []:
            cdf, mapping_func = calc_mapping_function(
                data_sep, reference_tmp)
            cdfs.append(cdf)
            mapping_funcs.append(mapping_func)
        else:
            cdf = cdfs[i]
            mapping_func = mapping_funcs[i]

        data_sep2 = match_cdf(data_sep, cdf, mapping_func)
        # ------------------------------------------------------

        data_sep2 = data_sep2.flatten()
        data_tmp2[:, i] = data_sep2

    data_per = np.reshape(data_tmp2, (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    return data_per, cdfs, mapping_funcs
