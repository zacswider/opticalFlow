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
imagePath = "/Users/bementmbp/Desktop/bzoptflow16-2.tif"    # full path to 1-channel image
imageStack=skio.imread(imagePath)   # reads image as ndArray
scale = 1                           # scale variable for displayed vector size; bigger value = smaller vector
step = 4                            # step size for vectors. Larger value = less vectors displayed
framesToSkip = 0                    # skip frames when comparing. Default is to compare consecutive frames (i.e., skip zero)
numBins = 64                        # number of bins for the polar histogram
start = 0                           # starting frame to compare, default is to start comparing with the first frame (index 0)
firstFrame = imageStack[start]                  # sets first image frame
secondFrame = imageStack[start+framesToSkip+1]  # sets second image frame

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

flow = calcFlow(frame1 = firstFrame, frame2 = secondFrame, pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 1) # calculates flow with default values
bins, widths, radius = calcVectors(flow)                    # calculates histogram bins, widths, and heights

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
startAx = fig.add_axes([.6, 0.825, 0.25, 0.05])     # rectangle of size [x0, y0, width, height]
startValues = np.linspace(0 ,imageStack.shape[0], imageStack.shape[0], endpoint=False)  # sets starting frame values
winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=20, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')                                 # window size slider parameters
nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')                                        # poly_n size slider parameters
sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')                                  # poly_sigma size slider parameters
startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=imageStack.shape[0], valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')       # start frame slider parameters
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
    f = int(startSlider.val)                # pulls value from start frame slider
    flow = calcFlow(frame1 = imageStack[f], frame2 = imageStack[f+framesToSkip+1], pyr = 0.5, lev = 3, win = w, it = 3, polN = n, polS = s, flag = 1)   #recalculates optical flow
    arrowsMerge.set_UVC(flow[::step, ::step, 0]*scale, flow[::step, ::step, 1]*-1*scale)    #updates arrow with new vector coordinates
    imgMerge.set_data(imageStack[f])        # updates image to display
    b, wi, radius = calcVectors(flow)       # recalculates bar heights from new vectors
    for i in range(len(radius)):            # iterates through all bars
        histBars[i].set_height(radius[i])   # sets new bar height
    ax2.set_ylim(top = np.max(radius))      # re-scales polar histogram y-axis
    fig.canvas.draw_idle()                  # re-draws the plot

winSlider.on_changed(update)                # calls update function if slider is changed
nSlider.on_changed(update)                  # calls update function if slider is changed
sSlider.on_changed(update)                  # calls update function if slider is changed
startSlider.on_changed(update)              # calls update function if slider is changed
plt.show()