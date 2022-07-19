"""
File: low_signal_detection.py

Author 1: J. Kuehne
Author 2: V. Kalchenko
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Product

Summary:
    This file implements a class which enables detection and visualisation of
    low strength signals in a sequence of images. The file was developed for
    use in in vivo imaging with bioluminescence, specifically to detect
    phtotons emmited on mice.
"""

import numpy as np
import cv2
from skimage import io
from sklearn.preprocessing import MinMaxScaler

RGB = 3

# transform features to [0, 1] -> default
min_max_scaler = MinMaxScaler()


def enhance_image(image, radius_denoising, radius_circle):
    """
    Summary:
        Enhances image with denoising and marking of the brightest spot

    Args:
        image (np.array): Image operations should be performed on
        radius_denoising (int): Radius used in denoising
        radius_circle (int): Radius of drawn circle

    Returns:
        image (np.array): Image with applied operation
    """
    # denoise
    image = cv2.fastNlMeansDenoisingColored(image, None, radius_denoising,
                                            radius_denoising, 7, 15)
    # convert to cv image grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # get brightest spot -> needs denoising
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
    # mark with circle
    cv2.circle(image, maxLoc, radius_circle, (255, 0, 0), 1)
    return image


def detect_signal(image: str, algorithm: list, do_enhance: bool,
                  radius_denoising: int = 20, radius_circle: int = 20):
    """
    Summary:
        Apply an algorithm to an image

    Args:
        image (str): Filename of image algorithm should be applied to
        algorithm (list): list of algorithms to apply
        do_enhance (bool): Whether to enhance image
        radius_denoising (int): radius used in denoising
        radius_circle (int): radius of drawn circle

    Returns:
        image (np.array): Enhanced image
    """
    img = io.imread(image)

    # take img dimensions and reshape
    if img.ndim == 3:
        frames, y, x = img.shape
        imr = np.reshape(img, (frames, x * y))
    else:
        frames, y, x, rgb = img.shape
        imr = np.reshape(img, (frames, x * y * RGB))

    # scale img into [0,1]
    imt = min_max_scaler.fit_transform(imr.T)

    output = algorithm.fit_transform(imt)

    output = min_max_scaler.fit_transform(output)

    if img.ndim == 3:
        image = np.reshape(output[:, 0], (y, x))
    else:
        image = np.reshape(output[:, 0], (y, x, RGB))

    # scale to image range
    image = np.array(image * 255).astype('uint8')

    # convert to cv image BGR if source was greyscale
    if img.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if do_enhance:
        image = enhance_image(image, radius_denoising, radius_circle)

    return image
