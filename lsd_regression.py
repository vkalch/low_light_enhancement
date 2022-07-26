"""
File lsd_regression.py

Author 1: J. Kuehne
Author 2: A. Balsam
Author 3: V. Kalchenko
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Product

Summary:
    This file uses unsupervised regression methods for low-light signal detection
"""

import cv2
import numpy as np
from skimage import io
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
RGB = 3


def detect_signal(image: str, algorithm):
    """
    Apply an algorithm to an image.

    :param image: Filename of image algorithm should be applied to
    :param algorithm: Algorithm to apply
    :return: Enhanced image
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

    return image
