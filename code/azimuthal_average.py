import numpy as np


def fftidx(n):
    """Essentially np.fftfreq, except that it returns integers
    """
    results = np.empty(n, dtype=int)
    nn = (n-1)//2 + 1
    results[:nn] = range(0, nn)
    results[nn:] = range(-(n//2), 0)
    return results

def azimuthalAverage(image):
    """Azimuthally average a 2D FFT image

    Notes:
    ------
    (zero frequency is on the top left, negative frequencies
    are on the bottom & right)
    """
    # Get distance to zero frequency
    fx = fftidx(image.shape[0])
    fy = fftidx(image.shape[1])
    r = np.hypot(fx[:,np.newaxis], fy[np.newaxis,:])

    # Sort image & r by distance from zero frequency
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Bin r to integers
    # Find all pixels within each radial bin
    r_int = r_sorted.astype(int)
    new_bin_idx = np.r_[0, np.where(r_int[1:] - r_int[:-1])[0] + 1]
    n_bin = np.r_[new_bin_idx[1:] - new_bin_idx[:-1], len(r_sorted)-new_bin_idx[-1]]

    # Sum image for each bin
    cs = np.r_[0, np.cumsum(i_sorted)]
    bintot = np.r_[
        cs[new_bin_idx[1:]]-cs[new_bin_idx[:-1]],
        cs[-1]-cs[new_bin_idx[-1]]]

    radial_prof = bintot / n_bin
    return radial_prof[1:], n_bin[1:]


def smooth_image(nx, ny):
    """Generate a smooth image"""
    image = np.zeros((nx, ny))

    # Add features
    for i in range(5):
        y,x = np.indices(image.shape)
        sigma = np.random.uniform(nx/10, nx/5)
        dx = x-np.random.uniform(0, nx)
        dy = y-np.random.uniform(0, ny)
        image += np.exp(-(dx**2+dy**2)/(2*sigma**2))

    # Return normalized image
    image /= image.max()
    return image

def sharp_image(nx, ny):
    """Generate an image with sharp edges"""
    image = np.zeros((nx, ny))

    # Add features
    for i in range(5):
        y,x = np.indices(image.shape)
        r = np.random.uniform(nx/10, nx/5)
        dx = x-np.random.uniform(0, nx)
        dy = y-np.random.uniform(0, ny)
        image += dx*dx + dy*dy < r*r

    # Return normalized image
    image /= image.max()
    return image

def add_noise(image, variance):
    """Adds Poissionian noise to image"""
    return image * np.random.poisson(variance, image.shape) / variance
