import cv2

colormaps = [
    ('Jet', cv2.COLORMAP_JET),
    ('HSV', cv2.COLORMAP_HSV),
    ('Rainbow', cv2.COLORMAP_RAINBOW),
    ('Grayscale', cv2.COLORMAP_BONE),
]


def get_colormaps():
    return colormaps


def get_colormap_by_name(colormap_name):
    for name, colormap in colormaps:
        if colormap_name == name:
            return colormap
