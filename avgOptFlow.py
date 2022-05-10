import time
import scipy
import colour
import cv2 as cv
import numpy as np
import pandas as pd
from tkinter import Tk
from pathlib import Path
from numpy.core.numeric import empty_like
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter.filedialog import askopenfilename

'''***** Default Parameters *****'''
#Tk().withdraw()                     # keeps the root tkinter window from appearing
#imagePath = askopenfilename()       # show an "Open" dialog box and return the path to the selected file
imagePath = "/Users/bementmbp/Desktop/210212_Live_SFC_Aegg_GFP-wGBD_mCh-Utr_Cntrl_E05-T01_Max.tif"    # full path to 1-channel image
savePath = Path(imagePath)
fileName = savePath.stem
saveDirectory = savePath.parent
imageStack = skio.imread(imagePath)   # reads image as ndArray
scale = 1                           # scale variable for displayed vector size; bigger value = smaller vector
step = 4                            # step size for vectors. Larger value = less vectors displayed
numBins = 64                        # number of bins for the polar histogram
brightness = 1000     #amplify brightness in draw_hsv display; default = 4
emptyList = []                                  # empty list to fill with bin values
for i in range(numBins+1):                        # builds empty list...
    emptyList.append(i*(6.28/numBins))          # full of equally radian values around a circle
useBins = np.array(emptyList)                   # and converts to np array to use in histogram calculation
goodWindowSize = 20
goodPolyN = 1
goodPolyS = 0.3
params = {"window":[goodWindowSize], "PolyN":[goodPolyN], "PolyS":[goodPolyS]}
paramDf = pd.DataFrame.from_dict(params)
txtSave = saveDirectory / (fileName + "_params.txt")
tfile = open(txtSave, 'a')
tfile.write(paramDf.to_string())
tfile.close()

'''***** Flow and Hist Functions *****'''
def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      # equation to calculate dense optical flow: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)                    # ndarray with the same shape as the image array

def draw_hsv(flow):                         #openCV function for drawing flow fields, with my modifications
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

def colour_wheel(samples=1024, clip_circle=True, method='Colour'): #https://stackoverflow.com/a/62544063/4812591
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

'''***** Array Mssaging *****'''
arraySize = (1, imageStack.shape[1], imageStack.shape[2])   # tuple of the correct shape to instantiate new empty arrays
allAngs = np.concatenate([np.zeros(arraySize) for i in range(imageStack.shape[0]-1)])                               # empty array to fill with angles from every image pair
allMags = np.concatenate([np.zeros(arraySize) for i in range(imageStack.shape[0]-1)])                               # empty array to fill with magnitudes from every image pair
firstConcat = np.concatenate([np.zeros(arraySize) for i in range(imageStack.shape[0]-1)])                           # empty array to fill with the first array from every flow array 
secondConcat = np.concatenate([np.zeros(arraySize) for i in range(imageStack.shape[0]-1)])                          # empty array to fill with the second array from every flow array 

for i in range(imageStack.shape[0]-1):                      #iterates through the frames in the image stack
    flow = calcFlow(imageStack[i], imageStack[i+1], pyr =0.5, lev = 3, win = goodWindowSize, it = 3, polN = goodPolyN, polS = goodPolyS, flag = 1)
    mags, angs = cv.cartToPolar(flow[:,:,0], flow[:,:,1]*-1, angleInDegrees = False)    # multiple by -1 to get the angles along the same axis as image
    firstDim = flow[:,:,0]                                                              # assigns first dimension of flow
    secDim = flow[:,:,1]                                                                # assigns second dimension of
    allAngs[i] = angs
    allMags[i] = mags
    firstConcat[i] = firstDim
    secondConcat[i] = secDim
    print(str(round((i+2)/imageStack.shape[0]*100, 1)) + "%" + " Finished with flow analysis")

print("Computing mean vectors...")
flatAngles = allAngs.flatten()                                                          # flattens the array into one dimension
flatMagnitudes = allMags.flatten()                                                      # flattens the array into one dimension
n, bins = np.histogram(flatAngles, bins=useBins, weights = flatMagnitudes)              # n is the counts and bins is ndarray of the equally spaced bins
widths = np.diff(bins)                                                                  # ndarray; width of each bin
radius = n                                                                              # sets the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height

firstAvg = np.mean(firstConcat, axis=0)                                              # averages the first frame values along axis 0 to calculate average flow
secondAvg = np.mean(secondConcat, axis=0)                                            # averages the second frame values along axis 0 to calculate average flow
firstAvg = np.expand_dims(firstAvg, axis=2)                                             # adds 3rd dimension
secondAvg = np.expand_dims(secondAvg, axis=2)                                           # adds 3rd dimension
avgFlow = np.concatenate((firstAvg, secondAvg), axis=2)                                 # concatenates the first and second frames of average flow into a 3d array

'''***** Plots and Sliders *****'''
fig = plt.figure()                                  # makes figure object
ax1 = plt.subplot(1, 2, 1)                          # axis object; left subplot
ax2 = plt.subplot(1, 2, 2, projection='polar')      # axis object; right subplot; uses polar coordinates
imgMerge = ax1.imshow(draw_hsv(avgFlow), aspect="equal") # matplotlib.image.AxesImage object; image of the first frame compared
newax = fig.add_axes([0.1, 0.8, 0.2, 0.2], anchor='NW')#, zorder=-1)
wheel = colour_wheel(samples=64)
rotatedWheel = scipy.ndimage.rotate(wheel, 180)
newax.imshow(rotatedWheel) #ROTATE 180
newax.axis('off')
ax1.set_xticks([])      # gets rid of x axis tick marks
ax1.set_yticks([])      # gets rid of y axis tick marks
#arrowsMerge = ax1.quiver(np.arange(0, avgFlow.shape[1], step), np.flipud(np.arange(avgFlow.shape[0]-1, -1, -step)),   # matplotlib.quiver.Quiver object; 
#                        avgFlow[::step, ::step, 0]*scale, avgFlow[::step, ::step, 1]*-1*scale, color="c")             # overlayed on axes image
histBars = ax2.bar(x = bins[:-1], height = radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=True, linewidth=1)   # Bar plot using histogram data
ax2.set_theta_zero_location("E")            # set the direction of the zero angle
ax2.set_yticks([])                          # gets rid of y axis tick marks
print("Done.")

imgSave = saveDirectory / (fileName + ".png")
plt.savefig(imgSave)
plt.show()
