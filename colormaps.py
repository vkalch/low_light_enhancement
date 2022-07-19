import cv2

colormaps = [
    ('Jet', cv2.COLORMAP_JET),
    ('Rainbow 1', cv2.COLORMAP_HSV),
    ('Rainbow 2', cv2.COLORMAP_RAINBOW),
    ('Grayscale', cv2.COLORMAP_BONE),
    ('Red', cv2.COLORMAP_HOT),
    ('Blue-Pink', cv2.COLORMAP_COOL)
]


def get_colormaps():
    return colormaps


def get_colormap_by_name(colormap_name):
    for name, colormap in colormaps:
        if colormap_name == name:
            return colormap
