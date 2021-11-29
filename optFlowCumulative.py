import cv2 as cv
import numpy as np
from tkinter import Tk
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter.filedialog import askopenfilename
#test
'''***** Default Parameters *****'''
#Tk().withdraw()                     # keeps the root tkinter window from appearing
#imagePath = askopenfilename()       # show an "Open" dialog box and return the path to the selected file
imagePath = "/Users/zacswider/Desktop/bzcropthick.tif"    # full path to 1-channel image
imageStack = skio.imread(imagePath)   # reads image as ndArray
print(imageStack.shape)
scale = 1                           # scale variable for displayed vector size; bigger value = smaller vector
step = 4                            # step size for vectors. Larger value = less vectors displayed
framesToSkip = 0                    # skip frames when comparing. Default is to compare consecutive frames (i.e., skip zero)
numBins = 64                        # number of bins for the polar histogram

'''***** Flow and Hist Functions *****'''
def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      # equation to calculate dense optical flow: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)                    # ndarray with the same shape as the image array

def calcVectors(flowArray):
    mags, angs = cv.cartToPolar(flowArray[:,:,0], flowArray[:,:,1]*(-1), angleInDegrees = False)  # converts flow to magnitude and angels; multiple vectors by -1 to get the angles along the same axis as image
    flatAngles = angs.flatten()                             # flattens the array into one dimension
    flatMagnitudes = mags.flatten()                         # flattens the array into one dimension
    n, bins = np.histogram(flatAngles, bins=numBins, weights = flatMagnitudes)      # n is the counts and bins is ndarray of the equally spaced bins between (-pi, pi)
    widths = np.diff(bins)                                  # ndarray; width of each bin
    radius = n                                              # sets the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height
    return(bins, widths, radius)

avgAngs = np.zeros((1, imageStack[0][1].size, imageStack[0][2].size))
avgMags = np.zeros((1, imageStack[0][1].size, imageStack[0][2].size))

for i in range(imageStack.shape[0]-1):
    flow = calcFlow(imageStack[i], imageStack[i+1], pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 1)
    mags, angs = cv.cartToPolar(flow[:,:,0], flow[:,:,1]*-1, angleInDegrees = False)  #multiple by -1 to get the angles along the same axis as image
    angs = np.expand_dims(angs, axis=0)
    mags = np.expand_dims(mags, axis=0)
    avgAngs = np.concatenate((avgAngs, angs))
    avgMags = np.concatenate((avgMags, mags))

flatAngles = avgAngs.flatten()                             # flattens the array into one dimension
flatMagnitudes = avgMags.flatten()                         # flattens the array into one dimension
n, bins = np.histogram(flatAngles, bins=numBins, weights = flatMagnitudes)      # n is the counts and bins is ndarray of the equally spaced bins between (-pi, pi)
widths = np.diff(bins)                                  # ndarray; width of each bin
radius = n                                              # sets the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height


'''***** Plots and Sliders *****'''

fig = plt.figure()                                  # makes figure object
ax1 = plt.subplot(1, 2, 1)                          # axis object; left subplot
ax2 = plt.subplot(1, 2, 2, projection='polar')      # axis object; right subplot; uses polar coordinates
imgMerge = ax1.imshow(imageStack[0], cmap = "copper", aspect = "equal") # matplotlib.image.AxesImage object; image of the first frame compared
ax1.set_xticks([])      # gets rid of x axis tick marks
ax1.set_yticks([])      # gets rid of y axis tick marks
#arrowsMerge = ax1.quiver(np.arange(0, flow.shape[1], step), np.flipud(np.arange(flow.shape[0]-1, -1, -step)),   # matplotlib.quiver.Quiver object; 
#                        flow[::step, ::step, 0]*scale, flow[::step, ::step, 1]*-1*scale, color="c")             # overlayed on axes image
histBars = ax2.bar(x = bins[:-1], height = radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=True, linewidth=1)   # Bar plot using histogram data
ax2.set_theta_zero_location("E")            # set the direction of the zero angle
ax2.set_yticks([])                          # gets rid of y axis tick marks

plt.show()


