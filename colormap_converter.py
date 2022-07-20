"""
File: colormap_converter.py

Author 1: D. Masek
Author 2: J. Kuehne
Author 3: A. Balsam
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Utility

Summary:
    This file takes a custom matplotlib colormap and saves it in an image file
    for later use with openCV independent of matplotlib.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

output_path = 'colormap.png'

# create custom discrete colormap, put string name here if native colormap
customcmap = colors.ListedColormap(['black', 'indigo', 'navy', 'royalblue', 
                                    'lightseagreen', 'green', 'limegreen', 
                                    'yellow', 'goldenrod', 'orange', 'red'])


"""
Summary:
    Takes matplotlib colormap and returns the transformed version in image
    format.

Args:
    cmap_name: Colormap object if custom, string with name if native
    
Returns: 
    colormap_image: colormap converted to image format
    
"""
def get_mpl_colormap(cmap_name):
    # get colormap if native
    cmap = plt.get_cmap(cmap_name)

    # initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    # transform
    colormap_image = color_range.reshape(256, 1, 3)
    
    return colormap_image


# save image
cv2.imwrite(output_path, get_mpl_colormap(customcmap))
