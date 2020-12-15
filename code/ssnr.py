import numpy as np


def fftidx(n):
    """Essentially `np.fftfreq` but returns integers"""
    results = np.empty(n, dtype=int)
    nn = (n-1)//2 + 1
    results[:nn] = range(0, nn)
    results[nn:] = range(-(n//2), 0)
    return results


def azimuthalAverage(image):
    """Azimuthally average a 2D FFT image
    
    Parameters
    ----------
    image : array_like
        Input image
        
    Returns
    -------
    radial_prof : 1D array
        Azimuthal average excluding zero frequency
    n_bin : 1D array
        Number of bins used in each average
        Can be used to find the original sum

    Notes
    -----
    * zero frequency is on the top left
    * negative frequencies are on the bottom & right
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
    n_bin = np.r_[new_bin_idx[1:] - new_bin_idx[:-1],
                len(r_sorted)-new_bin_idx[-1]]

    # Sum image for each bin
    cs = np.r_[0, np.cumsum(i_sorted)]
    bintot = np.r_[cs[new_bin_idx[1:]]-cs[new_bin_idx[:-1]],
                cs[-1]-cs[new_bin_idx[-1]]]

    radial_prof = bintot / n_bin
    return radial_prof[1:], n_bin[1:]


def SSNR_ring(image_list):
    """Compute the spectral signal-to-noise ratio
    averaged over equal-frequency rings"""
    ffts = [np.fft.fft2(image) / image.size for image in image_list]
    K = len(image_list)

    fftsum = np.sum(ffts, axis=0)
    num, _ = azimuthalAverage((fftsum*fftsum.conj()).real)
    den, _ = azimuthalAverage(
        np.sum([np.abs(ffts[i] - fftsum/K)**2 for i in range(K)], axis=0))

    # SSNR equation (Wikipedia) is for the combined average.
    # Divide by K to get the SNR for each individual image in the list.
    SSNR = num / (K/(K-1) * den) - 1
    return SSNR / K


def SSNR_full(image_list):
    """Compute the spectral signal-to-noise ratio
    averaged over the full images"""
    ffts = [np.fft.fft2(image) / image.size for image in image_list]
    for fft in ffts:
        fft[0, 0] = 0
    K = len(image_list)

    fftsum = np.sum(ffts, axis=0)
    num = np.sum((fftsum*fftsum.conj()).real)
    den = np.sum([np.abs(ffts[i] - fftsum/K)**2 for i in range(K)])

    # SSNR equation (Wikipedia) is for the combined average
    # Divide by K to get the SNR for each individual image in the list
    SSNR = num / (K/(K-1) * den) - 1
    return SSNR / K


def SNR_JOY(image):
    """Single image SNR method from Joy et al. (2002)
    
    Parameters
    ----------
    image : array_like
        Input image

    Returns
    -------
    SNR : float
        Computed signal to noise ratio
    """
    # Separate even and odd rows
    even = image[::2, :]
    oddd = image[1::2, :]

    # Compute SNR
    cov = np.mean((even.T - even.mean(axis=1)) *\
                (oddd.T - oddd.mean(axis=1)), axis=0)
    var = np.sqrt(np.var(even, ddof=1, axis=1) *\
                np.var(oddd, ddof=1, axis=1))
    Rn = cov / var
    snr = Rn / (1 - Rn)
    return snr.mean()


def split_image(image):
    """Split image along alternating horizontal and vertical scan lines"""
    im1 = image[0::2, 0::2]
    im2 = image[1::2, 0::2]
    im3 = image[0::2, 1::2]
    im4 = image[1::2, 1::2]
    images = [im1, im2, im3, im4]
    return images
