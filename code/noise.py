import math
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_image(nx, ny, N_features=500):
    """Generate a random image with shape (nx, ny)"""
    # Create features (Gaussian blobs)
    y, x = np.indices((nx, ny))
    x = np.repeat(x[:, :, np.newaxis], N_features, axis=2)
    y = np.repeat(y[:, :, np.newaxis], N_features, axis=2)
    sigma = np.random.uniform(nx/100, nx/50, N_features)
    dx = x - np.random.uniform(0, nx, N_features)
    dy = y - np.random.uniform(0, ny, N_features)
    blobs = np.exp(-(dx**2 + dy**2)/(2*sigma**2))
    image = np.sum(blobs, axis=2)

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
