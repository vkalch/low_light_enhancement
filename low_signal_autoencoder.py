"""
File: low_signal_detector_autoencoder.py

Author 1: A. Balsam
Author 2: J. Kuehne
Author 3: V. Kalchenko
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Product

Summary:
    This file implements a class which uses an autoencoder to detect low-light signals in a kinetic image.
    The user must pass a .tiff file to the encode_images() function, and can optionally select a customized output
    directory. This file has been separated from the other LSD functions because it must have the capability to train
    an autoencoder for each file.
"""

import logging
import statistics
import time
import shutil
import uuid

import imageio
import tensorflow as tf

from keras import layers
from keras.models import Model
from keras import optimizers

import numpy as np
import os
import tifffile
import skimage
import multipagetiff as mtif

from PIL import Image, ImageSequence

import matplotlib.pyplot as plt
from skimage import io


def populate_train_dir(train_dir, input_filepath):
    """
    Populates selected directory with frames of input file.

    :param train_dir: Directory to save training data in.
    :param input_filepath: Filepath of .tiff image to encode.

    :return: None, saves folder
    """
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    os.makedirs(train_dir)

    start = time.time()
    # Read .tiff file
    im = Image.open(input_filepath)

    frame_counter = 0
    for frame in ImageSequence.Iterator(im):
        frame_counter += 1
        frame.save(f'{train_dir}/frame_{frame_counter}.png')
    print(f"Saved file {input_filepath} in {round(time.time() - start, 2)}s")


def convolve(img):
    """
    Removes noise from image by averaging values of adjacent pixels.

    :param img: Image to denoise

    :returns: denoised: Denoised image
    """

    # Calculate mean pixel value (to save time on calculations later)
    img_mean = np.mean(img)

    # Find all black pixels
    x, y = np.where(
        (img[:, :] == 0)
    )

    # For each black pixel, set value to average value of neighbors (including diagonals)
    for pix in range(len(x)):
        neighbors = list()
        try:
            neighbors = [
                img[x[pix] + 1, y[pix]],
                img[x[pix] - 1, y[pix]],
                img[x[pix], y[pix] + 1],
                img[x[pix], y[pix] - 1],
                img[x[pix] + 1, y[pix] + 1],
                img[x[pix] - 1, y[pix] - 1],
                img[x[pix] - 1, y[pix] + 1],
                img[x[pix] + 1, y[pix] - 1]
            ]
            img[x[pix], y[pix]] = statistics.mean(neighbors)
        except IndexError:
            img[x[pix], y[pix]] = img_mean

    return img


class LowSignalAutoencoder:
    def __init__(self, image_size=56, num_epochs=200, dense_layer_neurons=2, show_loss_plot=False):
        """
        :param image_size: The size of the image to create.
        :param num_epochs: The number of epochs to run.
        :param dense_layer_neurons: The number of neurons to put in the smallest dense layer.
        :param show_loss_plot: Whether to show the loss plot of the image.
        """
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.dense_layer_neurons = dense_layer_neurons
        self.show_loss_plot = show_loss_plot

    def encode(self, input_filepath, train_dir=f"./static/training_data_{uuid.uuid4()}/",
               image_size=None, num_epochs=None, dense_layer_neurons=None, show_loss_plot=None):
        """
        Enhances a .tiff framesequence using an autoencoder and saves it to a file.

        :param input_filepath: Filepath of input .tiff file. This is the only parameter that must be set up by the user.
        :param train_dir: Temporary location to store training data.
        :param image_size: The size of the image to create.
        :param num_epochs: The number of epochs to run.
        :param dense_layer_neurons: The number of neurons to put in the smallest dense layer.
        :param show_loss_plot: Whether to show the loss plot of the image.
        :return: None, saves enhanced image to file
        """
        if image_size is None:
            image_size = self.image_size

        if num_epochs is None:
            num_epochs = self.num_epochs

        if dense_layer_neurons is None:
            dense_layer_neurons = self.dense_layer_neurons

        if show_loss_plot is None:
            show_loss_plot = self.show_loss_plot

        input_filename = input_filepath.split("/")[-1]
        input_dir_list = input_filepath.split("/")[:-1]
        base_dir = str()
        for i in input_dir_list:
            base_dir += i

        populate_train_dir(train_dir, input_filepath)

        # Get dataset from directory
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(f"{train_dir}", label_mode=None,
                                                                            image_size=(image_size, image_size),
                                                                            shuffle=False,
                                                                            color_mode='grayscale')

        # Normalize color values
        train_dataset = train_dataset.map(lambda x: x / 255)

        # Combine training and test datasets (if we get test data, replace the second
        # "train_dataset" with the test dataset)
        zipped_ds = tf.data.Dataset.zip((train_dataset, train_dataset))

        # Make autoencoder
        encoder_input = layers.Input(shape=(image_size, image_size, 1), name='img')
        flatten = layers.Flatten()(encoder_input)
        dense = layers.Dense(784, activation='relu')(flatten)
        dense = layers.Dense(512, activation='relu')(dense)
        dense = layers.Dense(128, activation='relu')(dense)
        encoder_output = layers.Dense(dense_layer_neurons, activation='linear')(dense)

        encoder = Model(encoder_input, encoder_output, name='encoder')

        dense = layers.Dense(128, activation='relu')(encoder_output)
        dense = layers.Dense(512, activation='relu')(dense)
        dense = layers.Dense(784, activation='relu')(dense)
        dense = layers.Dense(image_size * image_size, activation='relu')(dense)

        decoder_output = layers.Reshape((image_size, image_size, 1))(dense)

        decoder = Model(encoder_output, decoder_output, name='decoder')

        autoencoder = Model(encoder_input, decoder(encoder(encoder_input)))

        logging.info(autoencoder.summary())

        # Compile encoder using optimizer
        opt = optimizers.Adam(lr=0.001, decay=1e-12)
        autoencoder.compile(opt, loss='mse')

        # ============================================== Fitting =================================================
        history = autoencoder.fit(zipped_ds,
                                  epochs=num_epochs,
                                  batch_size=25,
                                  verbose=0,
                                  )
        # ========================================================================================================

        if show_loss_plot:
            # Plot loss
            loss = history.history['loss']
            plt.plot(range(num_epochs), loss, 'bo', label='Training loss')
            plt.title(f'Training loss {input_filename}')
            plt.legend()
            plt.show()

        image = tifffile.imread(input_filepath)
        resized_data = skimage.transform.resize(image, (image_size, image_size))

        img_np = resized_data.T

        enhanced_frames = list()
        raw_frames = list()

        i = 0
        for frame in img_np:
            i += 1
            ae_out = autoencoder.predict(frame.reshape(-1, image_size, image_size, 1)).reshape(image_size, image_size)
            raw_data = frame.reshape(-1, image_size, image_size, 1)

            raw_data = raw_data[0].reshape(image_size, image_size)

            enhanced_frames.append(ae_out)
            raw_frames.append(raw_data)

        # Create .tiff of encoded frames (for testing purposes)
        with tifffile.TiffWriter(f"{train_dir}/encoded_image.tiff") as tiff:
            for img in enhanced_frames:
                tiff.save(img)

        stack = mtif.Stack(np.array(enhanced_frames))
        plot = mtif.flatten(stack)
        plt.imsave(f"{train_dir}/color_coded_image.png", plot)

        # final_image = np.multiply(np.mean(enhanced_frames, axis=0), 10 ** 3)

        # Convert to grayscale
        img = Image.open(f"{train_dir}/color_coded_image.png").convert('L')
        img.save(f"{train_dir}/color_coded_image.png")

        final_image = io.imread(f"{train_dir}/color_coded_image.png")
        # Remove directory with training data for the next guy
        shutil.rmtree(train_dir)

        # This function no longer works since we're using png format
        final_image = convolve(final_image)
        return final_image


if __name__ == '__main__':
    INPUT_FILEPATH = ''
    a = LowSignalAutoencoder()
    a.encode(INPUT_FILEPATH)
