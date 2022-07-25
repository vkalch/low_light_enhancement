import logging

import cv2
import imageio
import tensorflow as tf

from keras import layers
from keras.models import Model
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import os
import statistics
import tifffile
import skimage

import time
import shutil

from PIL import Image, ImageSequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# for 0.2 noise, 200 epochs is optimal


# The path of a .tiff file in TIFF_DIR which will be processed by the autoencoder. If there are multiple noise levels,
# ensure that this image name exists for all of them
IMAGES_TO_PROCESS = ['sample1']

# The amount of noise to use -- make sure TIFF_DIR exists for this value
NOISE_LEVELS = ['0.1', '0.2', '1.1', '1.9']

# The square size (in pixels) of the images in the training dataset
IMAGE_SIZE = 56

# The number of epochs to run
NUM_EPOCHS = [100, 200, 200, 200]

DENSE_LAYER_NEURONS = [2, 2, 2, 2]

# Custom colormap for encoded images. Change this to whatever you like.
CUSTOM_COLORMAP = ListedColormap(['black', 'indigo', 'navy', 'royalblue', 'lightseagreen',
                                  'green', '#9CA84A', 'limegreen', '#E3F56C', 'yellow',
                                  'goldenrod', '#FFAE42', 'orange', '#ff6e11', 'red'])


def populate_train_dir(image_name):
    """
    Creates one folder in TRAIN_DIR representing a single .tiff image based on passed filename.

    :return: None, saves folder
    """

    if os.path.exists(f"{TRAIN_DIR}/{image_name}"):
        shutil.rmtree(f"{TRAIN_DIR}/{image_name}")

    os.makedirs(f"{TRAIN_DIR}/{image_name}")

    start = time.time()
    # Read tif file
    im = Image.open(f'{TIFF_DIR}/{image_name}.tiff')

    frame_counter = 0
    for frame in ImageSequence.Iterator(im):
        frame_counter += 1
        frame.save(f'{TRAIN_DIR}/{image_name}/frame_{frame_counter}.png')
    print(f"Saved file {image_name} in {round(time.time() - start, 2)}s")


def convolve(img):
    """
    Removes noise from image by averaging values of adjacent pixels.

    :param img: Image to denoise

    :return: Denoised image
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


def plot_figures(figures, nrows=1, ncols=1, cmap=CUSTOM_COLORMAP):
    """Plot a dictionary of figures.

    :param figures: <title, figure> dictionary
    :param ncols: number of columns of subplots wanted in the display
    :param nrows: number of rows of subplots wanted in the figure
    :param cmap: Colormap to use

    :return: None, plots selected images
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=cmap)
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional


def encode_tiff(f_name, output_name, cmap=CUSTOM_COLORMAP):
    """
    Runs a full .tiff file through the autoencoder

    :param f_name: Filename
    :param output_name: Output filename
    :param cmap: Colormap to use

    :return: None, saves encoded images
    """
    image = tifffile.imread(f_name)
    resized_data = skimage.transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    img_np = resized_data.T

    enhanced_frames = list()
    raw_frames = list()

    i = 0
    for frame in img_np:
        i += 1
        # print(f"Processing frame {i}. Shape={frame.shape}...")
        ae_out = autoencoder.predict(frame.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)).reshape(IMAGE_SIZE, IMAGE_SIZE)
        raw_data = frame.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        convolved = convolve(ae_out.reshape(IMAGE_SIZE, IMAGE_SIZE))

        raw_data = raw_data[0].reshape(IMAGE_SIZE, IMAGE_SIZE)
        figures = {
            "Original Image": raw_data,
            "Encoded Image": ae_out,
            # "Encoded + Denoised": convolved,
            # "Original": raw_data[0][0].reshape(IMAGE_SIZE,IMAGE_SIZE),
            # "Encoded": ae_out[0].reshape(IMAGE_SIZE,IMAGE_SIZE),
            # "E + D": convolved,
        }
        # plot_figures(figures, 1, 2, cmap)

        if not (os.path.exists(ENCODED_IMAGES_DIR)):
            os.mkdir(ENCODED_IMAGES_DIR)

        if not (os.path.exists(f"{ENCODED_IMAGES_DIR}/{output_name}/raw_data")):
            os.makedirs(f'{ENCODED_IMAGES_DIR}/{output_name}/raw_data')

        if not (os.path.exists(f"{ENCODED_IMAGES_DIR}/{output_name}/ae_out")):
            os.makedirs(f'{ENCODED_IMAGES_DIR}/{output_name}/ae_out')

        plt.imsave(f'{ENCODED_IMAGES_DIR}/{output_name}/raw_data/frame_{i}.png', raw_data, cmap=cmap)
        plt.imsave(f'{ENCODED_IMAGES_DIR}/{output_name}/ae_out/frame_{i}.png', ae_out, cmap=cmap)
        # plt.savefig(f'{ENCODED_IMAGES_DIR}/{output_name}/raw_data/frame_{i}.png', bbox_inches='tight', cmap=cmap)
        # plt.show()

        enhanced_frames.append(ae_out)
        raw_frames.append(raw_data)

    final_image = np.mean(enhanced_frames, axis=0)
    raw_data_mean = np.mean(raw_frames, axis=0)

    plt.imsave(f'{ENCODED_IMAGES_DIR}/{output_name}/enhanced_image_final_plt_color.png', final_image, cmap=cmap)
    plt.imsave(f'{ENCODED_IMAGES_DIR}/{output_name}/raw_image_mean_final_plt_color.png', raw_data_mean, cmap=cmap)
    plt.imsave(f'{ENCODED_IMAGES_DIR}/{output_name}/enhanced_image_final_plt_gray.png', final_image, cmap='gray')
    plt.imsave(f'{ENCODED_IMAGES_DIR}/{output_name}/raw_image_mean_final_plt_gray.png', raw_data_mean, cmap='gray')


for NOISE in NOISE_LEVELS:
    start_noise = time.time()
    print(f"Starting noise level {NOISE}...")

    # Directory which contains training data in .tiff format with convention "output_{i}.tiff" where "i" increases
    TIFF_DIR = f'/Users/avbalsam/Downloads/train_data_avi/merged_noise_{NOISE}'

    # Directory which will contain reformatted training data
    TRAIN_DIR = f'/Users/avbalsam/Downloads/train_data_avi/merged_noise_png_by_folder_{NOISE}_{IMAGE_SIZE}'

    # Directory which will contain encoded images
    ENCODED_IMAGES_DIR = f'/Users/avbalsam/Downloads/train_data_avi/encoded_images_by_folder_{NOISE}_{IMAGE_SIZE}'

    # Get the number of epochs to use for this noise level
    epochs = NUM_EPOCHS[NOISE_LEVELS.index(NOISE)]

    # Get the number of dense layer neurons to use on this noise level
    dense_layer_neurons = DENSE_LAYER_NEURONS[NOISE_LEVELS.index(NOISE)]

    for IMAGE_TO_PROCESS in IMAGES_TO_PROCESS:
        start_image = time.time()
        print(f"Starting image {IMAGE_TO_PROCESS}...")

        if populate_train_dir(IMAGE_TO_PROCESS) == "Training directory exists" and \
                input('Output dir already exists. Would you like to delete it? (Y/N)') == 'Y':
            shutil.rmtree(TRAIN_DIR)
            populate_train_dir(IMAGE_TO_PROCESS)

        # Get dataset from directory
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(f"{TRAIN_DIR}/{IMAGE_TO_PROCESS}", label_mode=None,
                                                                            image_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=False,
                                                                            color_mode='grayscale')

        # Normalize color values
        train_dataset = train_dataset.map(lambda x: x / 255)

        # Combine training and test datasets (if we get test data, replace the second
        # "train_dataset" with the test dataset)
        zipped_ds = tf.data.Dataset.zip((train_dataset, train_dataset))

        # Make autoencoder
        encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='img')
        flatten = layers.Flatten()(encoder_input)
        dense = layers.Dense(784, activation='relu')(flatten)
        dense = layers.Dense(512, activation='relu')(dense)
        dense = layers.Dense(128, activation='relu')(dense)
        encoder_output = layers.Dense(dense_layer_neurons, activation='linear')(dense)

        encoder = Model(encoder_input, encoder_output, name='encoder')

        dense = layers.Dense(128, activation='relu')(encoder_output)
        dense = layers.Dense(512, activation='relu')(dense)
        dense = layers.Dense(784, activation='relu')(dense)
        dense = layers.Dense(IMAGE_SIZE * IMAGE_SIZE, activation='relu')(dense)

        decoder_output = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(dense)

        decoder = Model(encoder_output, decoder_output, name='decoder')

        autoencoder = Model(encoder_input, decoder(encoder(encoder_input)))

        logging.info(autoencoder.summary())

        # Compile encoder using optimizer
        opt = optimizers.Adam(lr=0.001, decay=1e-12)
        autoencoder.compile(opt, loss='mse')

        # ============================================== Fitting =================================================
        history = autoencoder.fit(zipped_ds,
                                  epochs=epochs,
                                  batch_size=25,
                                  verbose=0,
                                  )
        # ========================================================================================================

        # Plot loss
        loss = history.history['loss']
        epochs = range(epochs)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training loss')
        plt.legend()
        # plt.show()

        encode_tiff(f'{TIFF_DIR}/{IMAGE_TO_PROCESS}.tiff', f"{IMAGE_TO_PROCESS}.tiff",
                    cmap=CUSTOM_COLORMAP)

        print(f"Finished image {IMAGE_TO_PROCESS} in {round(time.time()-start_image, 2)}s")
    print(f"Finished noise level {NOISE} in {round(time.time()-start_noise, 2)}s")
