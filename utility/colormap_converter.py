"""
File: colormap_converter.py

Author 1: D. Masek
Author 2: J. Kuehne
Author 3: A. Balsam
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Utility

Summary:
    This file takes a custom matplotlib utility and saves it in an image file
    for later use with openCV independent of matplotlib.
"""
import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

"""
The following is the custom colormap we used:

['black', 'indigo', 'navy', 'royalblue',
'lightseagreen', 'green', 'limegreen',
'yellow', 'goldenrod', 'orange', 'red']
"""


def make_cv_lut(cmap: str | list, output_path: str):
    """
    Saves opencv lookup table based on matplotlib colormap.

    :param cmap: List of color names or colormap object if custom, string with name if native
    :param output_path: Path to output lookup table
    :return: Colormap converted to image format
    """
    if type(cmap) == str:
        # get utility if native
        cmap = plt.get_cmap(cmap)
    elif type(cmap) == list:
        cmap = colors.ListedColormap(cmap)
    elif type(cmap) == colors.ListedColormap:
        cmap = cmap
    else:
        return None

    # initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    # transform
    colormap_image = color_range.reshape(256, 1, 3)

    cv2.imwrite(output_path, colormap_image)
