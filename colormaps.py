import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['black', 'indigo', 'navy', 'royalblue', 'lightseagreen',
                              'green', '#9CA84A', 'limegreen', '#E3F56C', 'yellow',
                              'goldenrod', '#FFAE42', 'orange', '#ff6e11', 'red'])

colormaps = [
    ('Default', custom_cmap),
    ('Jet', 'jet'),
    ('HSV', 'hsv'),
    ('Rainbow 1', 'rainbow'),
    ('Rainbow 2', 'gist_rainbow'),
    ('Grayscale', 'Greys'),
    ('Red', 'YlOrRd'),
    ('Blue-Red', 'bwr')
]


def get_colormaps():
    return colormaps


def get_colormap_by_name(colormap_name: str):
    """
    Get colormap by name.

    :param colormap_name: Name of colormap to get

    :return: Matplotlib usable colormap
    """
    for name, colormap in colormaps:
        if colormap_name == name:
            return colormap
