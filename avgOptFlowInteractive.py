import sys
import cv2 as cv
import numpy as np
from tkinter import Tk
from numpy.core.numeric import empty_like
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter.filedialog import askopenfilename

'''***** Default Parameters *****'''
Tk().withdraw()                     # keeps the root tkinter window from appearing
imagePath = askopenfilename()       # show an "Open" dialog box and return the path to the selected file
#imagePath = " "    # full path to 1-channel image
imageStack=skio.imread(imagePath)   # reads image as ndArray
scale = 1                           # scale variable for displayed vector size; bigger value = smaller vector
step = 4                            # step size for vectors. Larger value = less vectors displayed
numBins = 64                        # number of bins for the polar histogram
emptyList = []                                  # empty list to fill with bin values
for i in range(numBins+1):                        # builds empty list...
    emptyList.append(i*(6.28/numBins))          # full of equally radian values around a circle
useBins = np.array(emptyList)                   # and converts to np array to use in histogram calculation
goodWindowSize = 15
goodPolyN = 3
goodPolyS = 0.3

'''***** Quality Control *****'''
if imageStack.ndim > 3:
    print('The image has greater than three dimensions. Please load a single channel time lapse.')
    sys.exit()

'''***** Flow and Hist Functions *****'''
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
    flow = calcFlow(imageStack[i], imageStack[i+1], pyr =0.5, lev = 3, win = goodWindowSize, it = 3, polN = goodPolyN, polS = goodPolyS, flag = 1)
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
n, bins = np.histogram(flatAngles, bins=useBins, weights = flatMagnitudes)              # n is the counts and bins is ndarray of the equally spaced bins
widths = np.diff(bins)                                                                  # ndarray; width of each bin
radius = n                                                                              # sets the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height

firstAvg = np.average(firstConcat, axis=0)                                              # averages the first frame values along axis 0 to calculate average flow
secondAvg = np.average(secondConcat, axis=0)                                            # averages the second frame values along axis 0 to calculate average flow
firstAvg = np.expand_dims(firstAvg, axis=2)                                             # adds 3rd dimension
secondAvg = np.expand_dims(secondAvg, axis=2)                                           # adds 3rd dimension
avgFlow = np.concatenate((firstAvg, secondAvg), axis=2)                                 # concatenates the first and second frames of average flow into a 3d array

'''***** Plots and Sliders *****'''
fig = plt.figure()                                  # makes figure object
ax1 = plt.subplot(1, 2, 1)                          # axis object; left subplot
ax2 = plt.subplot(1, 2, 2, projection='polar')      # axis object; right subplot; uses polar coordinates
fig.subplots_adjust(top = 0.8)                      # sets spacing for top subplot
winAx = fig.add_axes([.185, 0.9, 0.25, 0.05])       # rectangle of size [x0, y0, width, height]
winValues = np.linspace(1,100,100)                  # sets window values 1-100
nAx = fig.add_axes([.185, 0.825, 0.25, 0.05])       # rectangle of size [x0, y0, width, height]
nValues = np.linspace(1,9,5)                        # sets poly_n values
sAx = fig.add_axes([.6, 0.9, 0.25, 0.05])           # rectangle of size [x0, y0, width, height]
sValues = np.linspace(0.1,2.0,20)                   # sets poly_sigma values
winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=goodWindowSize, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')                                 # window size slider parameters
nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=goodPolyN, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')                                        # poly_n size slider parameters
sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=goodPolyS, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')                                  # poly_sigma size slider parameters
imgMerge = ax1.imshow(imageStack[0], cmap = "copper", aspect = "equal") # matplotlib.image.AxesImage object; image of the first frame compared
ax1.set_xticks([])      # gets rid of x axis tick marks
ax1.set_yticks([])      # gets rid of y axis tick marks
arrowsMerge = ax1.quiver(np.arange(0, flow.shape[1], step), np.flipud(np.arange(flow.shape[0]-1, -1, -step)),   # matplotlib.quiver.Quiver object; 
                        flow[::step, ::step, 0]*scale, flow[::step, ::step, 1]*-1*scale, color="c")             # overlayed on axes image
histBars = ax2.bar(x = bins[:-1], height = radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=True, linewidth=1)   # Bar plot using histogram data
ax2.set_theta_zero_location("E")            # set the direction of the zero angle
ax2.set_yticks([])                          # gets rid of y axis tick marks

'''***** Update on Sliders *****'''
def update(val):                            # update function
    w = int(winSlider.val)                  # pulls value from win slider
    n = int(nSlider.val)                    # pulls value from poly_n slider
    s = sSlider.val                         # pulls value from poly_sigma slider
    
    arraySize = (1, imageStack.shape[1], imageStack.shape[2])   # tuple of the correct shape to instantiate new empty arrays
    allAngs = np.zeros(arraySize)                               # empty array to fill with angles from every image pair
    allMags = np.zeros(arraySize)                               # empty array to fill with magnitudes from every image pair
    firstConcat = np.zeros(arraySize)                           # empty array to fill with the first array from every flow array 
    secondConcat = np.zeros(arraySize)                          # empty array to fill with the second array from every flow array 

    for i in range(imageStack.shape[0]-1):                      #iterates through the frames in the image stack
        flow = calcFlow(imageStack[i], imageStack[i+1], pyr = 0.5, lev = 3, win = w, it = 3, polN = n, polS = s, flag = 1)   #recalculates optical flow
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
    print("average angle is:", str(np.average(flatAngles)*(180/np.pi)))
    flatMagnitudes = allMags.flatten()                                                      # flattens the array into one dimension
    print("average magnitude is:", str(np.average(flatMagnitudes)*(180/np.pi)))
    n, bins = np.histogram(flatAngles, bins=useBins, weights = flatMagnitudes)              # n is the counts and bins is ndarray of the equally spaced bins
    widths = np.diff(bins)                                                                  # ndarray; width of each bin
    radius = n                                                                              # sets the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height

    firstAvg = np.average(firstConcat, axis=0)                                              # averages the first frame values along axis 0 to calculate average flow
    secondAvg = np.average(secondConcat, axis=0)                                            # averages the second frame values along axis 0 to calculate average flow
    firstAvg = np.expand_dims(firstAvg, axis=2)                                             # adds 3rd dimension
    secondAvg = np.expand_dims(secondAvg, axis=2)                                           # adds 3rd dimension
    avgFlow = np.concatenate((firstAvg, secondAvg), axis=2)                                 # concatenates the first and second frames of average flow into a 3d array

    arrowsMerge.set_UVC(avgFlow[::step, ::step, 0]*scale, avgFlow[::step, ::step, 1]*-1*scale)    #updates arrow with new vector coordinates

    for i in range(len(radius)):            # iterates through all bars
        histBars[i].set_height(radius[i])   # sets new bar height
    ax2.set_ylim(top = np.max(radius))      # re-scales polar histogram y-axis
    fig.canvas.draw_idle()                  # re-draws the plot

winSlider.on_changed(update)                # calls update function if slider is changed
nSlider.on_changed(update)                  # calls update function if slider is changed
sSlider.on_changed(update)                  # calls update function if slider is changed
plt.show()