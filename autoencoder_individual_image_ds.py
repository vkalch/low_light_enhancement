import logging
import time
import shutil

import tensorflow as tf

from keras import layers
from keras.models import Model
from keras import optimizers

import numpy as np
import os
import tifffile
import skimage

from PIL import Image, ImageSequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# The base folder where training data and output files should be contained.
BASE_DIR = '/Users/avbalsam/Desktop/blender_animations'

# The path of a .tiff file in TIFF_DIR which will be processed by the autoencoder. If there are multiple noise levels,
# ensure that this image name exists for all of them
IMAGES_TO_PROCESS = ['sample1', 'sample2', 'sample3']

# The amount of noise to use -- make sure TIFF_DIR exists for this value
NOISE_LEVELS = ['0.1', '0.2', '0.3', '0.4', '0.6', '0.9', '1.9']

# The square size (in pixels) of the images in the training dataset.
# If there are fewer elements in this array than NOISE_LEVELS,
# the later NOISE_LEVELS will use the last element of this array.
IMAGE_SIZE = 56

# The number of epochs to run.
# If there are fewer elements in this array than NOISE_LEVELS,
# the later NOISE_LEVELS will use the last element of this array.
NUM_EPOCHS = [100, 200]

# The number of neurons to put in the dense layer.
# If there are fewer elements in this array than NOISE_LEVELS,
# the later NOISE_LEVELS will use the last element of this array.
DENSE_LAYER_NEURONS = [2]

# Custom colormap for encoded images. Change this to whatever you like.
CUSTOM_COLORMAP = ListedColormap(['black', 'indigo', 'navy', 'royalblue', 'lightseagreen',
                                  'green', '#9CA84A', 'limegreen', '#E3F56C', 'yellow',
                                  'goldenrod', '#FFAE42', 'orange', '#ff6e11', 'red'])


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


def encode_image(input_filepath, train_dir=None, output_dir=None, colormap='gray', image_size=56, num_epochs=200,
                 dense_layer_neurons=2, show_loss_plot=False):
    """
    Enhances a .tiff framesequence using an autoencoder and saves it to a file.

    :param input_filepath: Filepath of input .tiff file. This is the only parameter that must be set up by the user.
    :param train_dir: Directory in which training data will be stored. Defaults to the same directory as the raw image.
    :param output_dir: Directory in which output data will be stored. Defaults to same directory as raw image.
    :param colormap: The matplotlib colormap to apply to this image.
    :param image_size: The size of the image to create.
    :param num_epochs: The number of epochs to run.
    :param dense_layer_neurons: The number of neurons to put in the smallest dense layer.
    :param show_loss_plot: Whether to show the loss plot of the image.
    :return: None, saves enhanced image to file
    """
    input_filename = input_filepath.split("/")[-1]
    input_dir_list = input_filepath.split("/")[:-1]
    base_dir = str()
    for i in input_dir_list:
        base_dir += i

    # Set up directories
    if train_dir is None:
        train_dir = f"{base_dir}/training_data_{input_filename}"

    if not(os.path.exists(train_dir)):
        os.makedirs(train_dir)

    if output_dir is None:
        output_dir = f"{base_dir}/encoded_images_{input_filename}"

    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)

    if populate_train_dir(train_dir, input_filepath) == "Training directory exists" and \
            input('Output dir already exists. Would you like to delete it? (Y/N)') == 'Y':
        shutil.rmtree(train_dir)
        populate_train_dir(train_dir, input_filepath)

    # Get dataset from directory
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(f"{train_dir}", label_mode=None,
                                                                        image_size=(image_size, image_size), shuffle=False,
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

    # Plot loss
    loss = history.history['loss']
    plt.plot(range(num_epochs), loss, 'bo', label='Training loss')
    plt.title(f'Training loss {input_filename}')
    plt.legend()
    if show_loss_plot:
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

        if not (os.path.exists(ENCODED_IMAGES_DIR)):
            os.mkdir(ENCODED_IMAGES_DIR)

        if not (os.path.exists(f"{output_dir}/raw_data")):
            os.makedirs(f'{output_dir}/raw_data')

        if not (os.path.exists(f"{output_dir}/ae_out")):
            os.makedirs(f'{output_dir}/ae_out')

        plt.imsave(f'{output_dir}/raw_data/frame_{i}.png', raw_data, cmap=colormap)
        plt.imsave(f'{output_dir}/ae_out/frame_{i}.png', ae_out, cmap=colormap)

        enhanced_frames.append(ae_out)
        raw_frames.append(raw_data)

    final_image = np.mean(enhanced_frames, axis=0)
    raw_data_mean = np.mean(raw_frames, axis=0)

    plt.imsave(f'{output_dir}/enhanced_image_final_plt_color.png', final_image, cmap=colormap)
    plt.imsave(f'{output_dir}/raw_image_mean_final_plt_color.png', raw_data_mean, cmap=colormap)
    plt.imsave(f'{output_dir}/enhanced_image_final_plt_gray.png', final_image, cmap='gray')
    plt.imsave(f'{output_dir}/raw_image_mean_final_plt_gray.png', raw_data_mean, cmap='gray')


for NOISE in NOISE_LEVELS:
    for image_name in IMAGES_TO_PROCESS:
        start_noise = time.time()
        print(f"Starting noise level {NOISE}...")

        # Directory which contains training data in .tiff format with convention "output_{i}.tiff" where "i" increases
        TIFF_DIR = f'{BASE_DIR}/training_data/merged_noise_{NOISE}'

        # Directory which will contain reformatted training data
        TRAIN_DIR = f'{BASE_DIR}/training_data/merged_noise_png_by_folder_{NOISE}_{IMAGE_SIZE}'

        # Directory which will contain encoded images
        ENCODED_IMAGES_DIR = f'{BASE_DIR}/encoded_images_by_folder_{NOISE}_{IMAGE_SIZE}'

        for image_to_encode in IMAGES_TO_PROCESS:
            start_image = time.time()
            print(f"Starting image {image_to_encode}...")

            encode_image(input_filepath=f'{TIFF_DIR}/{image_to_encode}.tiff',
                         train_dir=f"{TRAIN_DIR}/{image_name}",
                         output_dir=f"{ENCODED_IMAGES_DIR}/{image_name}")

            print(f"Finished image {image_to_encode} in {round(time.time() - start_image, 2)}s")

        print(f"Finished noise level {NOISE} in {round(time.time()-start_noise, 2)}s")
