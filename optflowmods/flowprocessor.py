import sys
import cv2 as cv
import numpy as np
import scipy.ndimage as nd 
from tifffile import imread, TiffFile
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class FlowProcessor():

    def __init__(self, image_path, win_size, polyN, polyS, frame_skip, vect_skip, gauss_sigma):
        self.image_path = image_path
        self.win_size = win_size
        self.polyN = polyN
        self.polyS = polyS
        self.frame_skip = frame_skip
        self.vect_skip = vect_skip
        self.gauss_sigma = gauss_sigma
        self.image = imread(image_path)

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
        
        if self.num_frames < 2:
            print('Error: Image must have at least two frames in order to calculate flow')
            sys.exit()
        
        # convert image to 8 bit
        self.image = (((self.image - self.image.min()) / (self.image.max() - self.image.min()))*256).astype(np.uint8)
        
        # calculate number of sequential pairs
        self.num_pairs = self.num_frames - self.frame_skip - 1

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
        
        self.gauss = nd.gaussian_filter(self.image[:,0,:,:,:], sigma = (0,0,self.gauss_sigma,self.gauss_sigma))

    def calc_mean_flow(self):
        '''
        Calculates the mean optical flow for every sequential pair of frames in every channel
        in the image stack.
        Returns a numpy array of shape (num_frames, num_channels, 2)
        '''
        
        # empty array to fill with flow values
        self.flow_array = np.zeros((self.num_pairs, 
                                    self.num_channels, 
                                    self.gauss.shape[-2], 
                                    self.gauss.shape[-1], 
                                    2))

        # for every pair, calculates flow with specified values
        its = self.num_pairs * self.num_channels
        with tqdm(total = its, miniters = its/100) as pbar:
            pbar.set_description('Calculating flow of all pairs')
            for i in range(self.num_pairs):
                for j in range(self.num_channels):
                    frame1 = self.gauss[i,j,:,:]
                    frame2 = self.gauss[i + self.frame_skip + 1,j,:,:]
                    self.flow_array[i,j,:,:,:] = cv.calcOpticalFlowFarneback(prev = frame1, 
                                                                             next = frame2,
                                                                             flow = None,
                                                                             pyr_scale = 0.5,
                                                                             levels = 3,
                                                                             winsize = self.win_size,
                                                                             iterations = 3,
                                                                             poly_n = self.polyN, 
                                                                             poly_sigma = self.polyS, 
                                                                             flags = 1)
                    pbar.update(1)

        return(self.flow_array)

    def calc_regional_flow(self):
        '''
        For each array of calculated flow vectors, splits the vectors into its composite
        magnitude and angle components, then picks the collection of highest magnitude flow
        vectors by using either Otsu's threshold on the magnitudes array. 
        '''

        
        self.mags_array = np.zeros((self.num_pairs, 
                                    self.num_channels, 
                                    self.gauss.shape[-2], 
                                    self.gauss.shape[-1]))
        
        its = self.num_pairs * self.num_channels
        with tqdm(total = its, miniters = its/100) as pbar:
            pbar.set_description('Calculating regional magnitudes')
            for i in range(self.num_pairs):
                for j in range(self.num_channels):
                    
                    

                    
                    pbar.update(1)

        

