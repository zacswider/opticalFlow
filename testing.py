from optflowmods.flowgui import FlowGUI
from optflowmods.flowprocessor import FlowProcessor
from tqdm import tqdm
import sys
import os
from tifffile import imread
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im_path = '/Users/bementmbp/Desktop/Scripts/opticalFlow/test_data/200102_BZ_crop.tif'
window_size = 17
polyN_size = 3
polyS_size = 2
frames_to_skip = 15
vectors_to_skip = 8
gauss_sigma = 0


fp = FlowProcessor( image_path=im_path,
                    win_size = window_size,
                    polyN = polyN_size,
                    polyS = polyS_size,
                    frame_skip = frames_to_skip,
                    vect_skip = vectors_to_skip,
                    gauss_sigma = gauss_sigma)

flow_array = fp.calc_mean_flow()

mags_array = fp.calc_regional_flow()
#plt.show()



