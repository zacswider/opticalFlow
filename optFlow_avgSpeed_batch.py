'''
This script is for batch processing and plotting optical flow magnitude.
The user specifies a folder location with MAX projection time-series .tif files.
The script calculates the average magnitude of optical flow for each movie and generates color-coded plots.

USER MUST SET PARAMETERS FOR OPTICAL FLOW

'''

import cv2 as cv
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import os
from tkinter.filedialog import askdirectory
import datetime

goodWindowSize = 20                 #USER DEFINED
goodPolyN = 7                       #USER DEFINED
goodPolyS = 1.5                     #USER DEFINED

def set_up(): #select your initial directory
    targetWorkspace = askdirectory(title="Select your workspace")   #ask user for workspace with .tif files
    fileNames = [fname for fname in sorted(os.listdir(targetWorkspace)) if fname.endswith('.tif')] #get a list of all .tif files in location
    outputDir = os.path.join(targetWorkspace, "output")             #location for saved plots
    os.makedirs(outputDir, exist_ok=True)                           #make the directory for output plots
    return(targetWorkspace, fileNames, outputDir)

def calculate_flow(targetWorkspace, fileNames, outputDir): #calculate optical flow for each movie file
    my_dict = {}                       #empty dictionary for final values
    
    for i in range(0, len(fileNames)):
        '''***** Default Parameters *****'''
        imagePath = os.path.join(targetWorkspace, fileNames[i])    # full path to 1-channel image
        imageName = fileNames[i].rsplit(".",1)[0]                  #get the name of the file from the path (minus the suffix)

        print('Starting analysis of ', imageName)

        imageStack=skio.imread(imagePath)   # reads image as ndArray
        
        logParams = {"Window Size": goodWindowSize, "PolyN": goodPolyN, "PolyS": goodPolyS} #dict of parameters for log file

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

        my_dict[imageName] = avMag #append the file name and mags to a dictionary

    return(my_dict, logParams)    

def make_plots(dict, min, max, outputDir):
    print('Generating Plots...')
    for key, value in my_dict.items():
        plt.imshow(value, origin="lower", vmin = minValue, vmax = maxValue)
        #plt.colorbar() #add LUT color bar for each figure. Un-comment if you wish to do this
        plt.tight_layout()

        plotName = key + "_plot.png" #name for the fig
        savePath = os.path.join(outputDir, plotName) #path to output location

        plt.savefig(savePath) #saves fig
        plt.close() #closes fig

def plot_LUT(min, max, outputDir):
    print("Generating LUT plot...")
    number_of_ticks = 10
    gradient = np.linspace(1,256, 256)
    gradient = np.vstack((gradient, gradient))

    smallest = minValue
    largest = maxValue

    ticks = np.linspace(smallest, largest, number_of_ticks)
    tickLabels = [str(round(tick, 2)) for tick in ticks]

    fig, axs  = plt.subplots(nrows=1, figsize=(6.4, 1))         # figure and axis
    axs.set_xticks(ticks = np.linspace(1,250, number_of_ticks)) # this sets the x-tick values; evenly spaced ticks along the gradient that was plotted
    axs.set_xticklabels(labels = tickLabels)                    # and this sets the corresponding labels to display
    '''
    this would be:
    axs.set_xticks(ticks = np.linspace(1,250, number_of_ticks), labels = tickLabels)
    in matplotlib 3.5 and above
    '''
    axs.get_yaxis().set_visible(False)
    axs.imshow(gradient, aspect='auto')                         # show the image

    fig.tight_layout()
    figPath = os.path.join(outputDir, "LUT.png")
    fig.savefig(figPath)

def make_log(targetWorkspace, logParams):
    logPath = os.path.join(targetWorkspace, "log.txt")              # path to log file
    now = datetime.datetime.now()                                   # get current date and time
    logFile = open(logPath, "w")                                    # initiate text file
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     # write current date and time
    for key, value in logParams.items():                            # for each key:value pair in the parameter dictionary...
        logFile.write('%s: %s\n' % (key, value))                    # write pair to new line
    logFile.close()                                                 # close the file


### MAIN ####

targetWorkspace, fileNames, outputDir = set_up() #ask user for workspace with .tif files
my_dict, logParams = calculate_flow(targetWorkspace, fileNames, outputDir)
make_log(targetWorkspace, logParams)
maxValue = np.max([val for val in my_dict.values()]) #max of all values in the dataset
minValue = np.min([val for val in my_dict.values()]) #min of all values in the dataset
make_plots(my_dict, minValue, maxValue, outputDir)
plot_LUT(minValue, maxValue, outputDir)