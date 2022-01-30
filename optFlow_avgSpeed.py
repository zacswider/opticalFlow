import cv2 as cv
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

'''***** Default Parameters *****'''
imagePath = "/Users/bementmbp/Desktop/020718_Bg-EB1-H2B_05-prj-mrg-1.tif"    # full path to 1-channel image
imageStack=skio.imread(imagePath)   # reads image as ndArray
goodWindowSize = 25
goodPolyN = 5
goodPolyS = 1.1

'''***** Flow and Hist Functions *****'''
def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      # equation to calculate dense optical flow: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)                    # ndarray with the same shape as the image array

'''***** Array Massaging *****'''
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
avMag = np.mean(allMags, axis=0)                                                        # plot of average magnitudes
plt.imshow(avMag)
plt.tight_layout()
plt.show()