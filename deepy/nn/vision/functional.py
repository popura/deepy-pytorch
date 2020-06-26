def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _wsl_filter(img: torch.Tensor, L: torch.Tensor = None, smoothness=1, alpha=1.2, eps=1e-8):
    '''
    WLSFILTER Edge-preserving smoothing based on the weighted least squares(WLS) 
    optimization framework, as described in Farbman, Fattal, Lischinski, and
    Szeliski, "Edge-Preserving Decompositions for Multi-Scale Tone and Detail
    Manipulation", ACM Transactions on Graphics, 27(3), August 2008.
 
    Given an input image IN, we seek a new image OUT, which, on the one hand,
    is as close as possible to IN, and, at the same time, is as smooth as
    possible everywhere, except across significant gradients in L.
 
 
    Input arguments:
    ----------------
    img              Input image (2-D, double, N-by-M matrix). 
      
    smoothness          Balances between the data term and the smoothness
                    term. Increasing smoothness will produce smoother images.
                    Default value is 1.0
      
    alpha           Gives a degree of control over the affinities by non-
                    lineary scaling the gradients. Increasing alpha will
                    result in sharper preserved edges. Default value: 1.2
      
    L               Source image for the affinity matrix. Same dimensions
                    as the input image IN. Default: log(IN)
  
 
    Example 
    -------
    RGB = imread('peppers.png'); 
    I = double(rgb2gray(RGB));
    I = I./max(I(:));
    res = wlsFilter(I, 0.5);
    figure, imshow(I), figure, imshow(res)
    res = wlsFilter(I, 2, 2);
    figure, imshow(res)
    '''
    device = img.device
    img = cp.asarray(img.to('cpu').clone().detach().numpy())
    if L is None:
        L = cp.log(img + eps)
    else:
        L = cp.asarray(L.to('cpu').clone().detach().numpy())

    smallNum = 0.0001
    r, c = img.shape
    k = r * c

    # Compute affinities between adjacent pixels based on gradients of L
    dy = cp.diff(L, n=1, axis=0)
    dy = -smoothness / (cp.abs(dy) ** alpha + smallNum);
    dy = cp.pad(dy, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    dy = cp.ravel(dy)

    dx = cp.diff(L, n=1, axis=1) 
    dx = -smoothness / (cp.abs(dx) ** alpha + smallNum);
    dx = cp.pad(dx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    dx = cp.ravel(dx)


    # Construct a five-point spatially inhomogeneous Laplacian matrix
    B = cp.stack([dx, dy])
    d = cp.array([-r,-1])
    A = spdiags(B, d, k, k)

    e = dx
    w = cp.pad(dx, (r, 0), mode='constant', constant_values=0)
    w = w[0:-r]
    s = dy
    n = cp.pad(dy, (1, 0), mode='constant', constant_values=0)
    n = n[0:-1]

    D = 1 - (e + w + s + n)
    A = (A + A.T + spdiags(D, 0, k, k))

    # Solve
    out, *_ = lsqr(A, cp.ravel(img))
    out = cp.asnumpy(out)
    out = np.reshape(out, (r, c))
    return torch.from_numpy(out.astype(np.float32)).clone().to(device)