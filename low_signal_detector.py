"""
File: low_signal_detector.py

Author 1: J. Kuehne
Author 2: V. Kalchenko
Author 3: A. Balsam
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Product

Summary:
    This file implements a class which enables detection and visualisation of
    low strength signals in a sequence of images. The file was developed for
    use in in vivo imaging with bioluminescence, specifically to detect
    phtotons emitted on mice.
"""

import os
import time
import uuid

import cv2
import numpy as np
from skimage import io
from sklearn.preprocessing import MinMaxScaler

from algs import get_algorithm_name

RGB = 3

# transform features to [0, 1] -> default
min_max_scaler = MinMaxScaler()


class LowSignalDetector:
    """
    Class to handle image enhancement and store information about the images the user has enhanced.
    One instance of this class is created for every user.
    """

    def __init__(self, images_by_filename, algs, radius_denoising, radius_circle, colormap, do_enhance, output_folder):
        """
        Args:
            images_by_filename: Filenames of images to enhance
            algs: Algorithms to use
            radius_denoising: Denoising radius
            radius_circle: Radius of marker circle
            colormap: Colormap to use
            do_enhance: Whether to use opencv image enhancement
            output_folder: Directory where enhanced images should be outputted
        """
        self.ORIGINAL_IMAGES = list()
        self.ENHANCED_IMAGES = list()
        self.DOWNLOAD_PATHS_BY_IMAGE = list()
        self.ID = str(uuid.uuid4())

        self.images_by_filename = images_by_filename
        self.radius_denoising = radius_denoising
        self.radius_circle = radius_circle
        self.algs = algs
        self.colormap = colormap
        self.do_enhance = do_enhance
        self.output_folder = output_folder

    def get_id(self):
        return self.ID

    def run_algorithm(self):
        """
        Setter for this ImageEnhancer instance variables. Runs algorithms on selected images.

        Returns:
            None, sets self.ORIGINAL_IMAGES, self.ENHANCED_IMAGES, and self.DOWNLOAD_PATHS_BY_IMAGE
        """
        self.ORIGINAL_IMAGES = list()
        self.ENHANCED_IMAGES = list()
        self.DOWNLOAD_PATHS_BY_IMAGE = list()

        for img_num in range(len(self.images_by_filename)):
            self.ORIGINAL_IMAGES.append(self.images_by_filename[img_num])
            enhanced_image_by_alg = list()
            for abbr, alg in self.algs:
                start = time.time()
                enhanced_image = self.detect_signal(self.images_by_filename[img_num], alg)

                print(f"Finished doing {abbr} on {self.images_by_filename[img_num]} in {round(time.time() - start, 2)}s")

                path = os.path.join(self.output_folder, f'image{img_num}_{abbr}.png')
                enhanced_image = cv2.applyColorMap(enhanced_image, colormap=self.colormap)
                cv2.imwrite(path, enhanced_image)
                enhanced_image_by_alg.append({"alg_name": f"{get_algorithm_name(abbr)}", "filename": path})

            self.ENHANCED_IMAGES.append({"img_num": img_num, "enhanced_by_alg": enhanced_image_by_alg})

        paths = [[img['filename'] for img in enhanced_image['enhanced_by_alg']] for enhanced_image in
                 self.ENHANCED_IMAGES]

        for i in range(len(self.images_by_filename)):
            self.DOWNLOAD_PATHS_BY_IMAGE.append(paths[i])

        print("Finished running algorithms...")

    def enhance_image(self, image):
        """
        Summary:
            Enhances image with denoising and marking of the brightest spot

        Args:
            image (np.array): Image operations should be performed on

        Returns:
            image (np.array): Image with applied operation
        """
        # denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, self.radius_denoising,
                                                self.radius_denoising, 7, 15)
        # convert to cv image grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # get brightest spot -> needs denoising
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        # mark with circle
        cv2.circle(image, maxLoc, self.radius_circle, (255, 0, 0), 1)
        return image

    def detect_signal(self, image: str, algorithm: list):
        """
        Summary:
            Apply an algorithm to an image

        Args:
            image (str): Filename of image algorithm should be applied to
            algorithm: Algorithm to apply

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

        if self.do_enhance:
            image = self.enhance_image(image)

        return image
