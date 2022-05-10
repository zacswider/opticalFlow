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
imagePath = "/Users/bementmbp/Desktop/bzoptflow.tif"    # full path to 1-channel image
imageStack = skio.imread(imagePath)   # reads image as ndArray
scale = 1                           # scale variable for displayed vector size; bigger value = smaller vector
step = 4                            # step size for vectors. Larger value = less vectors displayed
framesToSkip = 0                    # skip frames when comparing. Default is to compare consecutive frames (i.e., skip zero)
numBins = 64                        # number of bins for the polar histogram

def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      # equation to calculate dense optical flow: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)                    # ndarray with the same shape as the image array

'''***** Array Mssaging *****'''
arraySize = (1, imageStack.shape[1], imageStack.shape[2])   # tuple of the correct shape to instantiate new empty arrays
allAngs = np.zeros(arraySize)                               # empty array to fill with angles from every image pair
allMags = np.zeros(arraySize)                               # empty array to fill with magnitudes from every image pair
firstConcat = np.zeros(arraySize)                           # empty array to fill with the first array from every flow array 
secondConcat = np.zeros(arraySize)                          # empty array to fill with the second array from every flow array 

for i in range(imageStack.shape[0]-1):                      #iterates through the frames in the image stack
    flow = calcFlow(imageStack[i], imageStack[i+1], pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 1)
    mags, angs = cv.cartToPolar(flow[:,:,0], flow[:,:,1]*-1, angleInDegrees = False)    # multiple by -1 to get the angles along the same axis as image
    firstDim = flow[:,:,0]                                                              # assigns first dimension of flow
    secDim = flow[:,:,1]                                                                # assigns second dimension of
    angs = np.expand_dims(angs, axis=0)                                                 # adds 3rd dimension
    mags = np.expand_dims(mags, axis=0)                                                 # adds 3rd dimension
    firstDim = np.expand_dims(firstDim, axis=0)                                         # adds 3rd dimension
    secDim = np.expand_dims(secDim, axis=0)                                             # adds 3rd dimension
    allAngs = np.concatenate((allAngs, angs))                                           # concatenates onto growing array
    allMags = np.concatenate((allMags, mags))                                           # concatenates onto growing array
    firstConcat = np.concatenate((firstConcat, firstDim))                               # concatenates onto growing array
    secondConcat = np.concatenate((secondConcat, secDim))                               # concatenates onto growing array

allAngs = np.delete(allAngs, obj=0, axis=0)                                             # deletes the initial array full of zeros
allMags = np.delete(allMags, obj=0, axis=0)                                             # deletes the initial array full of zeros
firstConcat = np.delete(firstConcat, obj=0, axis=0)                                     # deletes the initial array full of zeros
secondConcat = np.delete(secondConcat, obj=0, axis=0)                                   # deletes the initial array full of zeros

flatAngles = allAngs.flatten()                                                          # flattens the array into one dimension
flatMagnitudes = allMags.flatten()                                                      # flattens the array into one dimension
n, bins = np.histogram(flatAngles, bins=numBins, weights = flatMagnitudes)              # n is the counts and bins is ndarray of the equally spaced bins
widths = np.diff(bins)                                                                  # ndarray; width of each bin
radius = n                                                                              # sets the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height

firstAvg = np.average(firstConcat, axis=0)                                              # averages the first frame values along axis 0 to calculate average flow
secondAvg = np.average(secondConcat, axis=0)                                            # averages the second frame values along axis 0 to calculate average flow
firstAvg = np.expand_dims(firstAvg, axis=2)                                             # adds 3rd dimension
secondAvg = np.expand_dims(secondAvg, axis=2)                                           # adds 3rd dimension
avgFlow = np.concatenate((firstAvg, secondAvg), axis=2)                                 # concatenates the first and second frames of average flow into a 3d array

'''***** Plots *****'''
fig = plt.figure()                                                      # makes figure object
ax1 = plt.subplot(1, 2, 1)                                              # axis object; left subplot
ax2 = plt.subplot(1, 2, 2, projection='polar')                          # axis object; right subplot; uses polar coordinates
imgMerge = ax1.imshow(imageStack[0], cmap = "copper", aspect = "equal") # matplotlib.image.AxesImage object; image of the first frame compared
ax1.set_xticks([])                                                      # gets rid of x axis tick marks
ax1.set_yticks([])                                                      # gets rid of y axis tick marks
arrowsMerge = ax1.quiver(np.arange(0, avgFlow.shape[1], step), np.flipud(np.arange(avgFlow.shape[0]-1, -1, -step)),   # matplotlib.quiver.Quiver object; 
                        avgFlow[::step, ::step, 0]*scale, avgFlow[::step, ::step, 1]*-1*scale, color="c")             # overlayed on axes image
histBars = ax2.bar(x = bins[:-1], height = radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=True, linewidth=1)   # Bar plot using histogram data
ax2.set_theta_zero_location("E")            # set the direction of the zero angle
ax2.set_yticks([])                          # gets rid of y axis tick marks

plt.show()




