import cv2
import numpy as np


def validate_image(image):
    if image is None:
        raise ValueError("Image could not be loaded.")
    if image.size == 0:
        raise ValueError("Image is empty.")
    return True


def grayscale(image):
    validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_filter(image, kernel_size=5):
    validate_image(image)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_filter(image, kernel_size=5):
    validate_image(image)
    return cv2.medianBlur(image, kernel_size)


def histogram_equalization(image):
    gray = grayscale(image)
    return cv2.equalizeHist(gray)


def fourier_transform(image):
    gray = grayscale(image)

    spectrum = np.fft.fft2(gray)
    shifted = np.fft.fftshift(spectrum)

    magnitude = 20 * np.log(np.abs(shifted) + 1)

    return cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


def canny_edges(image, low=100, high=200):
    gray = grayscale(image)
    return cv2.Canny(gray, low, high)
