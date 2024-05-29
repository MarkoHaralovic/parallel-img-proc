import numpy as np
import cupy as cp
from scipy.fftpack import fft2, ifft2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.registration import phase_cross_correlation
from cupyx.scipy.signal import convolve2d
from skimage.restoration import inpaint
import pywt
from skimage.util import view_as_windows
from skimage.color import rgb2gray
from skimage.restoration import inpaint
import time

def perform_fft(image):
    if isinstance(image, cp.ndarray):
        return cp.fft.fft2(image)
    return fft2(image)

def perform_ifft(image):
    if isinstance(image, cp.ndarray):
        return cp.fft.ifft2(image)
    return ifft2(image)

def perform_wavelet_transform(image, wavelet='db1', level=1):
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)  
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def perform_convolution(image, kernel):
    if image.ndim == 3 and kernel.ndim == 2:
        if isinstance(image, cp.ndarray):
            kernel = cp.asarray(kernel)
            return cp.stack([convolve2d(channel, kernel) for channel in cp.rollaxis(image, -1)])
        else:
            return np.stack([convolve2d(channel.get(), kernel.get()) for channel in np.rollaxis(image, -1)])
    elif image.ndim == 2:
        if isinstance(image, cp.ndarray):
            kernel = cp.asarray(kernel)
            return convolve2d(image, kernel)
        return convolve2d(image, kernel)
    else:
        raise ValueError("Unexpected image dimensions")

def perform_nl_means_denoising(image):
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))
    return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True)


def perform_image_registration(image1, image2):
    if isinstance(image1, cp.ndarray):
        image1 = cp.asnumpy(image1)
    if isinstance(image2, cp.ndarray):
        image2 = cp.asnumpy(image2)
    shift, error, diffphase = phase_cross_correlation(image1, image2)
    return shift, error, diffphase

def sliding_window_detection(image, window_size, step_size, classifier):
    """
    Perform object detection using a sliding window approach.
    
    Parameters:
    - image: Input image.
    - window_size: Size of the sliding window (height, width).
    - step_size: Number of pixels to move the window in each step.
    - classifier: A function that takes a window and returns a classification result.
    
    Returns:
    - detections: List of tuples (x, y, window, classification).
    """
    
    if len(image.shape) == 3:
        print(type(image))
        image = rgb2gray(image)
    
    windows = view_as_windows(image, window_size, step=step_size)
    detections = []

    for y in range(windows.shape[0]):
        for x in range(windows.shape[1]):
            window = windows[y, x]
            classification = classifier(window)
            if classification: 
                detections.append((x * step_size, y * step_size, window, classification))
    
    return detections

def perform_inpainting(image, mask):
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    if isinstance(mask, cp.ndarray):
        mask = cp.asnumpy(mask)
    return inpaint.inpaint_biharmonic(image, mask)

def mean_and_std(image):
    channels_means, channels_std = [], []
    for channel in range(image.shape[2]):
        channels_means.append(np.mean(image[:,:,channel]))
        channels_std.append(np.std(image[:,:,channel]))
    return np.mean(image), np.std(image)

def time_synchronize():
   return time.time()
   