"""
File: low_signal_detector.py

Author 1: J. Kuehne
Author 2: A. Balsam
Author 3: V. Kalchenko
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
import logging

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
from sklearn.preprocessing import MinMaxScaler
import imageio.v3 as iio

from algs import get_algorithm_name, is_autoencoder
import lsd_regression as lsr

# transform features to [0, 1] -> default
min_max_scaler = MinMaxScaler()


class LowSignalDetector:
    """
    Class to handle image enhancement and store information about the images the user has enhanced.
    One instance of this class is created for every user.
    """
    def __init__(self, images_by_filename: list, algs: list, radius_denoising: int, radius_circle: int,
                 colormap: str, do_enhance: bool, show_circle: bool, autoencoder_image_size: int,
                 autoencoder_num_epochs: int, autoencoder_dense_layer_neurons: int, output_folder: str):
        """
        :param images_by_filename: Filenames of images to enhance
        :param algs: Algorithms to use
        :param radius_denoising: Denoising radius
        :param radius_circle: Radius of marker circle
        :param colormap: Colormap to use
        :param do_enhance: Whether to use opencv image enhancement
        :param show_circle: Whether to show marker circle
        :param autoencoder_image_size: Size of images used for autoencoder
        :param autoencoder_num_epochs: Number of epochs for autoencoder to run
        :param autoencoder_dense_layer_neurons: Number of neurons in the dense layer
        :param output_folder: Directory where enhanced images should be outputted
        """
        self.ORIGINAL_IMAGES = list()
        self.ENHANCED_IMAGES = list()
        self.DOWNLOAD_PATHS_BY_IMAGE = list()
        self.ID = str(uuid.uuid4())

        self.ERR = None

        self.images_by_filename = images_by_filename
        self.radius_denoising = radius_denoising
        self.radius_circle = radius_circle
        self.autoencoder_image_size = autoencoder_image_size
        self.autoencoder_num_epochs = autoencoder_num_epochs
        self.autoencoder_dense_layer_neurons = autoencoder_dense_layer_neurons
        self.algs = algs
        self.colormap = colormap

        self.do_enhance = do_enhance
        self.show_circle = show_circle
        self.output_folder = output_folder

    def __str__(self):
        return (f"Low Signal Detector {self.get_id()}\nAlgs: {self.algs}\nImages: {self.images_by_filename}\n"
                f"Circle Radius: {self.radius_circle}\nDenoising Radius: {self.radius_denoising}\n"
                f"Colormap: {self.colormap}\nEnhance: {self.do_enhance}\nOutput Folder: {self.output_folder}")

    def get_id(self):
        return self.ID

    def run_algorithms(self):
        """
        Setter for this ImageEnhancer instance variables. Runs algorithms on selected images.

        :return: None, sets self.ORIGINAL_IMAGES, self.ENHANCED_IMAGES, and self.DOWNLOAD_PATHS_BY_IMAGE
        """
        logging.info(f"Running run_algorithm() on:\n{str(self)}")

        self.ERR = None
        self.ORIGINAL_IMAGES = list()
        self.ENHANCED_IMAGES = list()
        self.DOWNLOAD_PATHS_BY_IMAGE = list()

        start_algs = time.time()

        for img_num in range(len(self.images_by_filename)):
            # Save gif files to display (for comparison's sake)
            gif_path = f"{self.images_by_filename[img_num].split('.')[0]}.gif"

            try:
                frames = iio.imread(self.images_by_filename[img_num], index=None)
            except (OSError, ValueError):
                logging.error(f"A corrupted or unreadable file was uploaded by the user: "
                              f"{self.images_by_filename.pop(img_num)}. Skipping this file...")
                img_num -= 1
                continue

            iio.imwrite(gif_path, frames)
            self.ORIGINAL_IMAGES.append(gif_path)

            enhanced_image_by_alg = list()
            for abbr, alg in self.algs:
                start = time.time()
                logging.info(f"Starting {abbr} on {self.images_by_filename[img_num]}...")

                if is_autoencoder(alg):
                    start = time.time()
                    print("Using autoencoder...")
                    enhanced_image = alg.encode(self.images_by_filename[img_num],
                                                image_size=self.autoencoder_image_size,
                                                num_epochs=self.autoencoder_num_epochs,
                                                dense_layer_neurons=self.autoencoder_dense_layer_neurons
                                                )
                    print(f"Finished with autoencoder in {round(time.time()-start, 2)}s")
                else:
                    print("Using regression methods...")
                    enhanced_image = lsr.detect_signal(self.images_by_filename[img_num], alg)
                    print("Finished using regression methods...")

                path = os.path.join(self.output_folder, f'image{img_num}_{abbr}.png')

                # Convert to grayscale so matplotlib can apply colormap
                plt.imsave(path, enhanced_image)
                enhanced_image = Image.open(path).convert('L')
                enhanced_image.save(path)

                enhanced_image = io.imread(path)
                plt.imsave(path, enhanced_image, cmap=self.colormap)

                if self.do_enhance or self.show_circle:
                    enhanced_image_cv = cv2.imread(path)
                    if self.do_enhance:
                        enhanced_image_cv = self.denoise_image(enhanced_image_cv)

                    # convert to cv image grayscale
                    enhanced_image_cv = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2GRAY)

                    if self.show_circle:
                        enhanced_image_cv = self.mark_circle(enhanced_image_cv)

                    cv2.imwrite(path, enhanced_image_cv)

                enhanced_image_by_alg.append({"alg_name": f"{get_algorithm_name(abbr)}", "filename": path})

                logging.info(
                    f"Finished {abbr} on {self.images_by_filename[img_num]} in {round(time.time() - start, 2)}s")

            self.ENHANCED_IMAGES.append({"img_num": img_num, "enhanced_by_alg": enhanced_image_by_alg})

        paths = [[img['filename'] for img in enhanced_image['enhanced_by_alg']] for enhanced_image in
                 self.ENHANCED_IMAGES]

        logging.info(f"Enhanced Images: {self.ENHANCED_IMAGES}")

        if self.ENHANCED_IMAGES == list():
            self.ERR = "User did not upload any readable image files."

        for i in range(len(self.images_by_filename)):
            self.DOWNLOAD_PATHS_BY_IMAGE.append(paths[i])

        logging.info(f"Finished running algorithms. Total time: {round(time.time() - start_algs, 2)}s")

    def denoise_image(self, image: np.array):
        """
        Enhance an image using opencv.

        :param image: Image operations should be performed on
        :return: Enhanced image
        """
        # denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, self.radius_denoising,
                                                self.radius_denoising, 7, 15)

        return image

    def mark_circle(self, image):
        """
        Marks brightest spot on image with a circle.

        :param image: Image to mark
        :return: Marked image
        """
        # get brightest spot -> needs denoising
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        # mark with circle
        cv2.circle(image, maxLoc, self.radius_circle, (255, 0, 0), 1)

        return image
