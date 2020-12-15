import math
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_image(nx, ny):
    """Generate a random image with shape (nx, ny)"""
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


def white_noise(variance, shape):
    """Generate white noise"""
    return np.random.normal(0, np.sqrt(variance), shape)


def nonwhite_noise(variance, shape):
    """Generate non-white (correlated) noise"""
    white = np.random.normal(0, 1, shape)
    correlated = gaussian_filter(white, .4)
    return math.sqrt(variance/correlated.var()) * correlated
