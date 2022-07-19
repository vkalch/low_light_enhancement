"""
File: noise_merge.py

Author 1: A. Balsam
Author 2: J. Kuehne
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Utility

Summary:
    This file takes separate frames of a sequence of tif files and merges them.
    Additionaly it adds varying amounts of noise. 
    The strength of poisson / shotnoise
    is constant, the amount of gaussian noise is variable.
"""

import os
import tifftools
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageSequence
import shutil
from natsort import natsorted
import time
import math

# files will get stored in this folder too
# directory must consist out of subfolders, one sequence per subfolder
FILES_DIR = '/home/jonas/Downloads/blender'
OUTPUT_NAME = 'sample'
# name of the subfolder where the merged files will get stored
OUTPUT_DIR = 'merged_noise'


NOISE_AMOUNT_POISSON = 1
# specs gaussian noise
MIN_NOISE = 0.0
MAX_NOISE = 2.0
INCREMENT = 0.1


"""
Summary:
    Merges .tiff files in the specified directory in alphabetical order and 
        returns the merged file.

Args:
    files_dir (str): Directory which contains (only) all .tiff files to merge
    
Returns: 
    merged: Merged .tiff file
"""
def merge_tiffs(files_dir:str):
    # get tif list
    try:
        tif_list = natsorted(os.listdir(files_dir))
    except:
        return

    merged = tifftools.read_tiff(f'{files_dir}/{tif_list[0]}')
    for tif in range(1, len(tif_list)):
        tif2 = tifftools.read_tiff(f'{files_dir}/{tif_list[tif]}')
        # Add input2 to input1
        merged['ifds'].extend(tif2['ifds'])

    return merged


"""
Summary:
    Adds gaussian and poisson noise to full .tiff file and saves to output_path

Args:
    filename (str): The image to modify
        output_path (str): Path to output file 
        (including filename and .tiff extension).
    noise_amount_gaussian: 
        The amount of noise to add for gaussian noise (see scikit docs)
    
Returns:
    void, saves images
"""
def add_noise(filename, noise_amount_gaussian=0.005, intensity=-0.15):
    # get img
    im = Image.open(filename)

    # add new folder to contain temp noisy images, deleted later on
    if not(os.path.exists('./noisy')):
        os.mkdir('./noisy')

    # iterate through frames
    i = 0
    for frame in ImageSequence.Iterator(im):
        i += 1
        # convert PIL Image to ndarray
        img = np.asarray(frame.convert('L')).astype(np.uint8)

        # first iteration (noise_amount =  0) should add no noise
        if not(math.isclose(noise_amount_gaussian, 0.0)):
            # poisson noise -> signal dependent, thus image scaled
            img = (img / 255) * NOISE_AMOUNT_POISSON
            # apply noise
            img = np.random.poisson(img)
            # scale back
            img = ((255 * img) / NOISE_AMOUNT_POISSON).astype(np.uint8)
            # add gaussian noise as not signal dependent
            img = random_noise(img, mode='gaussian', var=noise_amount_gaussian, 
                               mean=intensity)
            # scale back (library method converts to [0, 1]) and convert
            img = (255 * img).astype(np.uint8)
        
        # get image from array
        img = Image.fromarray(img)
        # save to temp directory
        img.save(f'./noisy/frame_{i}.tiff')
    
    # merge
    img = merge_tiffs(f'./noisy')
    
    # save merged file and delete temp directory
    if os.path.exists(filename):
        os.remove(filename)
    tifftools.write_tiff(img, filename)
    shutil.rmtree('./noisy')
    return

"""
Summary: 
    Prepares directories for file output. 
    Before running, make sure all .tiff files are grouped into subfolders 
    with format sample{i}.

Args:
    files_dir (str): Directory in which input image folders are contained and 
        to which files will be outputted
    output_name (str): Template for output file names. A number will be added 
        to the end of this to get output filename.
    noise_mode: which noise mode to use
    
Returns:
    void
"""
def format_output_images(files_dir:str, output_name:str, output_dir:str, noise):
    i = 0
    # take output directory
    output_dir = f'{output_dir}_{noise}'
    while os.path.exists(f'{files_dir}/{output_dir}'):
        output_dir = f'{output_dir}(1)'
    
    # create directory
    os.mkdir(f'{files_dir}/{output_dir}')
    
    # create outputs mostly
    while os.path.exists(f'{files_dir}/sample{i}'):
        start = time.time()
        print(f"Iterating through sample{i}...")
        output_path = f'{files_dir}/{output_dir}/{output_name}{i}.tiff'
        merged = merge_tiffs(f'{files_dir}/sample{i}')

        if os.path.exists(output_path):
            if input("Output file already exists. " +
                     "Do you want to delete it? (Y/N) ") == 'N':
                return
            else:
                os.remove(output_path)
        tifftools.write_tiff(merged, output_path)

        if noise:
            # Adds noise to image that was just outputted
            add_noise(output_path, noise)
            #add_gaussian_noise(output_path, noise)
        print(f"Finished iterating through sample{i}. Time elapsed: " +
              "{round(time.time()-start,2)}s")
        i += 1
    return


# main, increases noise from MIN_NOISE to MAX_NOISE in INCREMENT steps
for noise in np.arange(MIN_NOISE, MAX_NOISE, INCREMENT):
    format_output_images(FILES_DIR, OUTPUT_NAME, OUTPUT_DIR, round(noise, 1))
