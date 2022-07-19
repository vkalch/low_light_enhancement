"""
File: luciferin_spline_visualisation.py

Author 1: J. Kuehne
Author 2: Avi Balsam
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Utility

Summary:
    This file visualises the spline interpolation used for simulating the
    kinetic curved as seen in in vivo imaging with bioluminescence.
    The spline is used for data generation for ML models in blender.
    Corresponding blender file: data_gen_discrete.blend
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import random

# how many plots to create (randomised parameters)
NUM_PLOTS = 100

#################### PASTE CODE FROM BLENDER HERE -> PART 1 ####################

# specs
# size of images
img_size = 128
# how many frames
num_frames = 100
# how many different samples
num_samples = 500
# interval possible light intensity blob at peak
min_intensity_blob = 0.2
max_intensity_blob = 0.4
# interval possible light intensity mouse at start
min_init_intensity_mouse = 6500
max_init_intensity_mouse = 7500
# interval possible light intensity mouse at end
min_end_intensity_mouse = 4500
max_end_intensity_mouse = 5500
# scaling
original_scaling_mouse = 0.420
original_scaling_blob = 4.20
min_scaling_mouse = 0.9
max_scaling_mouse = 1.1
min_scaling_blob = 0.7
max_scaling_blob = 1.3
# shift
max_shift_mouse_x = 2
max_shift_mouse_y = 2
max_shift_blob_x = 1.5
max_shift_blob_y = 5

################################################################################

# create plots
for i in range(NUM_PLOTS):


#################### PASTE CODE FROM BLENDER HERE -> PART 2 ####################

    # frame with highest light intensity blob -> randomized around 0.3 * num_frames +/- 0.05 * num_frames
    highest_intensity_frame = int((num_frames * 0.3) + (random.uniform(-1.0, 1.0) * 0.05 * num_frames))
    # initial values
    # mouse at start
    init_intensity_mouse = random.uniform(min_init_intensity_mouse, max_init_intensity_mouse)
    # mouse at end
    end_intensity_mouse = random.uniform(min_end_intensity_mouse, max_end_intensity_mouse)
    # max blob
    peak_intensity_blob = random.uniform(min_intensity_blob, max_intensity_blob)
        
    # create spline for light emission of blob
    # x - values
    x = [0,
         highest_intensity_frame / 100,
         highest_intensity_frame / 4, 
         highest_intensity_frame / 2, 
         highest_intensity_frame, 
         (3/2) * highest_intensity_frame, 
         (num_frames + highest_intensity_frame) / 2, 
         (3/4) * num_frames + highest_intensity_frame / 4, 
         num_frames - 1,
         num_frames]
         
    # y - values
    y = [0.0, 
         0.0,
         peak_intensity_blob / 5, 
         (4/5) * peak_intensity_blob, 
         peak_intensity_blob, 
         (7/10) * peak_intensity_blob, 
         (3 / 10) * peak_intensity_blob, 
         (1 / 10) * peak_intensity_blob, 
         0.0,
         0.0]

################################################################################

    # create spline object and specs for plot
    n = len(y)
    spline = interpolate.splrep(x, y, s=0)
    xfit = np.arange(0, num_frames, np.pi/50)
    yfit = interpolate.splev(xfit, spline, der=0)
    # plot
    plt.plot(x, y, 'ro')
    plt.plot(xfit, yfit,'b')
    plt.plot(xfit, yfit)
    plt.title("Light emission luciferin")
    plt.show()
