import os
import cv2
import numpy as np

"""
This script plays specified movie overlaid with optical flow vectors and hsv flow field.
Modified from script written by Nicolai Høirup Nielsen at https://github.com/niconielsen32/ComputerVision/tree/master/opticalFlow
Can be easily modified to accept live frames from a webcame or other accessory camera
"""

magnitude = 5       #amplify arrow size in draw_flow display; default = 1
brightness = 32     #amplify brightness in draw_hsv display; default = 4
stepSize = 8        #density of arrows to draw (1 = max, 16 = sparse)
framesToSkip = 0    #number of frames to skip before comparing

def draw_flow(img, flow, step=stepSize):    #openCV function for drawing flow fields, with my modifications
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x-fx*magnitude, y-fy*magnitude]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    return img_bgr

def draw_hsv(flow):                         #openCV function for drawing flow fields, with my modifications
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*brightness, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def findWorkspace(directory, prompt):                                                       #accepts a starting directory and a prompt for the GUI
    #targetWorkspace = askdirectory(initialdir=directory, message=prompt)                    #opens prompt asking for folder, keep commented to default to baseDirectory
    targetWorkspace = directory                                                            #comment this out later if you want a GUI
    filelist = [fname for fname in os.listdir(targetWorkspace) if fname.endswith('.tif')]   #Makes a list of file names that end with .tif
    return(targetWorkspace, filelist)                                                       #returns the folder path and list of file names
baseDirectory = "/Users/bementmbp/Desktop/testImages/eb1_crop"                              #specifies base directory
directory, fileNames = findWorkspace(baseDirectory, "PLEASE SELECT YOUR SOURCE WORKSPACE")  #string object describing the file path, list object containing all file names ending with .tif
fileNames.sort()                                                                            #sorts the file names so they are indexed properly

i = 0                                                                   #starting index
prev = cv2.imread(directory + "/" + fileNames[i])                       #opens the first frame in the folder
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)                       #makes sure that it is grayscale
while True:                                                             #loops until canceled
    img = cv2.imread(directory + "/" + fileNames[i+1+framesToSkip])     #opens the next frame, depending on what skips is set to    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        #makes sure that it is grayscale
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 1, 
                            poly_n = 3, poly_sigma = .5, flags = 1)     #calculates flow
    prevgray = gray                                                     #sets most recently opened image as the previous image for next loop
    if i == len(fileNames)-2-framesToSkip:                              #makes sure that we never go out of index, accounting for skipped frames
        i=0                                                             #sets i back to 0 if we do
    else:                                                               #otherwise...
        i = i+1                                                         #advances i by one value (frame)
    cv2.imshow('flow', draw_flow(gray, flow))                           #calls draw_flow
    cv2.imshow('flow HSV', draw_hsv(flow))                              #alls draw_hsv
    key = cv2.waitKey(62)                                             #wait time in ms between frames
    if key == ord('q'):
        break                                                           #breaks the loop if you hit q

cv2.destroyAllWindows()                                                 #closes all windows
