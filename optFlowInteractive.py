import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path
import skimage.io as skio 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

scale = 1           # scale variable for displayed vector size; bigger value = smaller vector
step = 4           # step size for vectors. Larger value = less vectors displayed
framesToSkip = 0    # skip frames when comparing. Default is to compare consecutive frames (i.e., skip zero)
numBins = 64        # number of bins for the polar plot
imagePath = "/Users/zacswider/Desktop/eb1_crop.tif"    #path to 1-channel image
imageStack=skio.imread(imagePath)           # reads image as ndArra
start = 0
firstFrame = imageStack[start]                  # sets first image frame
secondFrame = imageStack[start+framesToSkip+1]  # sets second image frame

def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      # equation to calculate dense optical flow
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)    # ndarray with the same shape as the image array

vectors = calcFlow(frame1 = firstFrame, frame2 = secondFrame, pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 0)
mags, angs = cv.cartToPolar(vectors[:,:,0], vectors[:,:,1]*(-1), angleInDegrees = False)  # converts flow to magnitude and angels; multiple vectors by -1 to get the angles along the same axis as image
angles = angs.flatten()                             # flattens the array into one dimension
magnitudes = mags.flatten()                         # flattens the array into one dimension
#angles = angles[angles != 0.]                       # filters out all occurances of 0.
fig = plt.figure()                                  # makes figure object
ax1 = plt.subplot(1, 2, 1)                          # axis object; left subplot
ax4 = plt.subplot(1, 2, 2, projection='polar')      # axis object; right subplot; uses polar coordinates
#ax5 = plt.subplot(2, 1, 2)
fig.subplots_adjust(top = 0.8)                      # sets spacing for top subplot
#ax5.hist(angles)
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
ax1.set_xticks([])                                                      # gets rid of x axis tick marks
ax1.set_yticks([])                                                      # gets rid of y axis tick marks
arrowsMerge = ax1.quiver(np.arange(0, vectors.shape[1], step), np.flipud(np.arange(vectors.shape[0]-1, -1, -step)), 
                        vectors[::step, ::step, 0]*scale, vectors[::step, ::step, 1]*-1*scale, color="c")               # matplotlib.quiver.Quiver object; overlayed on axes image

#x = (angles+np.pi) % (2*np.pi) - np.pi          # converts angles from radians to (-pi, pi)
n, bins = np.histogram(angles, bins=numBins, weights = magnitudes)         # n is the counts and bins is ndarray of the equally spaced bins between (-pi, pi)
widths = np.diff(bins)                          # ndarray; width of each bin
area = n / angles.size                               # fraction of the total counts in each bin; area to assign each bin
#radius = (area/np.pi) ** .5                     # Calculate corresponding bin radius
radius = n
patches = ax4.bar(x = bins[:-1], height = radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=True, linewidth=1)   # Plot data on ax
ax4.set_theta_zero_location("E")                         # Set the direction of the zero angle
#ax4.set_theta_direction(-1)
ax4.set_yticks([])

def update(val):
    w = int(winSlider.val)
    n = int(nSlider.val)
    s = sSlider.val
    f = int(startSlider.val)
    newVectors = calcFlow(frame1 = imageStack[f], frame2 = imageStack[f+framesToSkip+1], pyr = 0.5, lev = 3, win = w, it = 3, polN = n, polS = s, flag = 1)
    newMags, newAngs = cv.cartToPolar(newVectors[:,:,0], newVectors[:,:,1]*-1, angleInDegrees = False) #multiple by -1 to get the angles along the same axis as image
    newAngles = newAngs.flatten()
    newMagnitudes = newMags.flatten()
    arrowsMerge.set_UVC(newVectors[::step, ::step, 0]*scale, newVectors[::step, ::step, 1]*-1*scale)
    imgMerge.set_data(imageStack[f])  

    newN, newBins = np.histogram(newAngles, bins=numBins, weights = newMagnitudes)                # Bin data and record counts
    newRadius = newN
    print("Recalculating:")
    print("max = " + str(np.max(newRadius)))
    for i in range(len(newRadius)):
        patches[i].set_height(newRadius[i])
    ax4.set_ylim(top = np.max(newRadius))
    fig.canvas.draw_idle()      #re-draws the plot

winSlider.on_changed(update)   #calls update function if slider is changed
nSlider.on_changed(update)   #calls update function if slider is changed
sSlider.on_changed(update)   #calls update function if slider is changed
startSlider.on_changed(update)   #calls update function if slider is changed
plt.show()