import sys
import cv2 as cv
import numpy as np
import scipy.ndimage as nd 
from tifffile import imread, TiffFile
import matplotlib.pyplot as plt

class FlowProcessor():

    def __init__(self, image_path, frames_to_skip, gauss_sigma):
        self.image_path = image_path
        self.gauss_sigma = gauss_sigma
        self.framesToSkip = frames_to_skip
        self.image = imread(self.image_path)

        # standardize image dimensions
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)
        self.num_slices = metadata.get('slices', 1)
        self.num_frames = metadata.get('frames', 1)
        self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])

        # max project image stack if num_slices > 1
        if self.num_slices > 1:
            print(f'Max projecting image stack')
            self.image = np.max(self.image, axis = 1)
            self.num_slices = 1
            self.image = self.image.reshape(self.num_frames, 
                                            self.num_slices, 
                                            self.num_channels, 
                                            self.image.shape[-2], 
                                            self.image.shape[-1])
        
        self.gauss = nd.gaussian_filter(self.image, sigma = (0,0,0,self.gauss_sigma,self.gauss_sigma))

    def get_flow(self):
        '''
        
        '''

        def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      
            # https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
            flow = cv.calcOpticalFlowFarneback(prev = np.invert(np.array(frame1)), 
                                               next = np.invert(np.array(frame2)), 
                                               flow = None,
                                               pyr_scale = pyr, 
                                               levels = lev,
                                               winsize = win, 
                                               iterations = it,
                                               poly_n = polN, 
                                               poly_sigma = polS, 
                                               flags = flag)

        # calculates flow with default values
        flow = calcFlow(frame1 = self.gauss[0],
                        frame2 = self.gauss[1],
                        pyr =0.5, 
                        lev = 3, 
                        win = 20, 
                        it = 3, 
                        polN = 7, 
                        polS = 1.5, 
                        flag = 1) 

        return(flow)