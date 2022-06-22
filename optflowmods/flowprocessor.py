import sys
import cv2 as cv
import numpy as np
import scipy.ndimage as nd 
from tifffile import imread, TiffFile
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from skimage.filters import threshold_otsu
import colour
import skimage

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
        
        # normalize to uint8, if necessary
        if not self.image.dtype == np.uint8:
            
            self.image = (((self.image - self.image.min()) / (self.image.max() - self.image.min()))*255).astype(np.uint8)
        
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
        
        # get rid of the empty z axis
        self.image = self.image[:,0,:,:,:]

        self.gauss = nd.gaussian_filter(self.image[:,:,:,:], sigma = (0,0,self.gauss_sigma,self.gauss_sigma))

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
        self.flow_array = self.flow_array[:,:,self.win_size:-self.win_size,self.win_size:-self.win_size,:]
        return(self.flow_array)

    def calc_regional_flow(self):
        '''
        For each array of calculated flow vectors, splits the vectors into its composite
        magnitude and angle components, then picks the collection of highest magnitude flow
        vectors by using either Otsu's threshold on the magnitudes array. 
        '''
        # empty array to fill with magnitude values for each pair and each channel
        self.mags_array = np.zeros((self.num_pairs, 
                                    self.num_channels, 
                                    self.flow_array.shape[-3], 
                                    self.flow_array.shape[-2]))
        
        # need a dictionary because we don't know how many channels there will be
        self.masked_mags_array = {}
        
        its = self.num_pairs * self.num_channels
        with tqdm(total = its, miniters = its/100) as pbar:
            pbar.set_description('Calculating regional magnitudes')
            for j in range(self.num_channels):

                self.masked_mags_array[f'Ch{j + 1}'] = []
                for i in range(self.num_pairs):
                    mags, _ = cv.cartToPolar(self.flow_array[i,j,:,:,0], 
                                                self.flow_array[i,j,:,:,1]*(-1),    # inverse to keep same orientation as image
                                                angleInDegrees = False)
                    self.mags_array[i,j,:,:] = mags

                    # Otsu threshold magnitude array
                    mags_thresh = threshold_otsu(mags)
                    mags_mask = mags > mags_thresh
                    self.masked_mags_array[f'Ch{j + 1}'].append(mags[mags_mask])

                    pbar.update(1)
        
        return self.mags_array, self.masked_mags_array

    def plot_summary(self):
        '''
        Summarize the results of the flow calculation.
        Returns a dictionary containing a figure for each channel.
        '''
        def draw_hsv(flow, corr=0):
            '''
            OpenCV function for converting flow to HSV with my modifications.
            Corr is a correction factor for the brightness of the displayed values
            '''
            brightness = 1000 / (corr + 1)
            h, w = flow.shape[:2]
            fx, fy = flow[:,:,0], flow[:,:,1]
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx*fx+fy*fy)
            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[...,0] = ang*(180/np.pi/2)
            hsv[...,1] = 255
            hsv[...,2] = np.minimum(v*brightness, 255)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            return bgr

        def colour_wheel(samples=1024, clip_circle=True, method='Colour'): 
            '''
            Returns a ndarray of RGB values for a colour wheel.
            Modified from stackoverflow.com/a/62544063/4812591
            '''
            xx, yy = np.meshgrid(
                np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))
            S = np.sqrt(xx ** 2 + yy ** 2)    
            H = (np.arctan2(xx, yy) + np.pi) / (np.pi * 2)
            HSV = colour.utilities.tstack([H, S, np.ones(H.shape)])
            RGB = colour.HSV_to_RGB(HSV)
            if clip_circle == True:
                RGB[S > 1] = 0
                A = np.where(S > 1, 0, 1)
            else:
                A = np.ones(S.shape)
            if method.lower()== 'matplotlib':
                RGB = colour.utilities.orient(RGB, '90 CW')
            elif method.lower()== 'nuke':
                RGB = colour.utilities.orient(RGB, 'Flip')
                RGB = colour.utilities.orient(RGB, '90 CW')
            R, G, B = colour.utilities.tsplit(RGB)
            return colour.utilities.tstack([R, G, B, A])

        def get_figure(channel_num):
            '''
            Returns a figure summarizing some optical flow properties in random frames as well
            as over the course of the whole time lapse.
            Returns a dictionary with keys f'Ch{i + 1}' and plots as values.
            '''
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

            all_masked_mags = np.concatenate(self.masked_mags_array[f'Ch{channel_num + 1}'])
            ax1.hist(self.mags_array[:,channel_num,:,:].ravel(), bins = 100, color = 'tab:blue', label = 'total')
            ax1.hist(all_masked_mags, bins = 100, color = 'tab:orange', label = 'masked')
            ax1.set_title('Histogram of magnitudes')
            ax1.set_xlabel('Magnitude')
            ax1.set_ylabel('Count')
            ax1.legend(loc = 'upper right')

            rand_index = np.random.randint(0, self.num_pairs-1)
            ax2.imshow(self.image[rand_index,channel_num,:,:], cmap = 'gray')
            ax2.set_title(f'Frame {rand_index}')

            example_thresh = threshold_otsu(self.mags_array[rand_index,channel_num,:,:])
            example_mask = self.mags_array[rand_index,channel_num,:,:] > example_thresh
            outlines = skimage.segmentation.find_boundaries(example_mask)
            merge = np.zeros((example_mask.shape[0], example_mask.shape[1], 3))
            merge[:,:,0] = outlines
            merge[:,:,1] = self.mags_array[rand_index,channel_num,:,:]
            merge[:,:,2] = outlines
            
            #ax3.imshow(self.mags_array[rand_index,channel_num,:,:])
            ax3.imshow(merge)
            ax3.set_title(f'Frame {rand_index} flow magnitude')

            data = [self.mags_array.ravel(), all_masked_mags]
            labels = ['Total', 'Masked']
            boxes = ax4.boxplot(data, labels = labels)
            for i in range(len(data)):
                x, y = boxes['medians'][i].get_xydata()[1]
                data_mean = str(round(np.mean(data[i]), 2))
                data_std = str(round(np.std(data[i]), 2))
                text = f'Mean: {data_mean}\nStd: {data_std}'
                ax4.text(x, y, text, fontsize = 'xx-small', color = '#cc7000')  
            
            ax4.set_title('Boxplot of magnitudes')
            ax4.set_ylabel('Magnitude (pixels/frame')


            mean_flow_array = np.mean(self.flow_array[:,channel_num,:,:,:], axis = 0)
            empty = np.zeros((self.flow_array.shape[-3], self.flow_array.shape[-2]))
            step = self.vect_skip
            # matplotlib.quiver.Quiver object overlayed on axes image
            ax5.imshow(empty, cmap = 'gray')
            ax5.quiver(np.arange(0, mean_flow_array.shape[1], step), 
                                np.flipud(np.arange(mean_flow_array.shape[0]-1, -1, -step)),   
                                mean_flow_array[::step, ::step, 0], 
                                mean_flow_array[::step, ::step, 1]*-1, 
                                color="c")
            ax5.set_title('Mean flow vectors')

            ax6.imshow(draw_hsv(np.mean(self.flow_array[:,channel_num,:,:,:], axis = 0), corr = 0))
            # get and set color wheel
            pos = ax6.get_position()
            newax = fig.add_axes([pos.x0, pos.y0, 0.05, 0.05], anchor='NW')#, zorder=-1)
            wheel = colour_wheel(samples=64)
            rotatedWheel = nd.rotate(wheel, 180)
            newax.imshow(rotatedWheel) #ROTATE 180
            newax.axis('off')
            ax6.set_title('Mean flow (HSV)')

            for ax in [ax3, ax2, ax5, ax6]:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.subplots_adjust(wspace=0.2, hspace=0.3)
            

            plt.close(fig)
            return fig

        plot_dict = {}
        for i in range(self.num_channels):
            plot_dict[f'Ch{i + 1}'] = get_figure(i)
        
        return plot_dict




    
        

        





