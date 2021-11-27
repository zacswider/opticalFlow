import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

frames = list(Path('/Users/bementmbp/Desktop/testImages/eb1_crop').iterdir())
frames.sort()
step = 4
framesToCompare = 1
firstFrame = Image.open(frames[0])
secondFrame = Image.open(frames[0+framesToCompare])
scale = 1 #bigger = smaller vector

def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)

vectors = calcFlow(frame1 = firstFrame, frame2 = secondFrame, pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 0)
mags, angs = cv.cartToPolar(vectors[:,:,0], vectors[:,:,1]*-1, angleInDegrees = False) #multiple by -1 to get the angles along the same axis as image
angles = angs.flatten()
magnitudes = mags.flatten()

fig = plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4, projection='polar')
fig.subplots_adjust(top = 0.8)

winAx = fig.add_axes([.125, 0.9, 0.175, 0.05]) #rectangle of size [x0, y0, width, height]
winValues = np.linspace(1,100,100)
nAx = fig.add_axes([.125, 0.825, 0.175, 0.05]) #rectangle of size [x0, y0, width, height]
nValues = np.linspace(1,9,5)
sAx = fig.add_axes([.5, 0.9, 0.175, 0.05]) #rectangle of size [x0, y0, width, height]
sValues = np.linspace(0.1,2.0,20)
startAx = fig.add_axes([.5, 0.825, 0.175, 0.05]) #rectangle of size [x0, y0, width, height]
startValues = np.linspace(0,len(frames),len(frames))

winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=20, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')
nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')
sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')
startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=len(frames), valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')

img = ax1.imshow(Image.open(frames[0]), cmap = "gray", aspect = "auto")
arrows = ax2.quiver(np.arange(0, vectors.shape[1], step), np.flipud(np.arange(vectors.shape[0]-1, -1, -step)), 
                      vectors[::step, ::step, 0]*scale, vectors[::step, ::step, 1]*-1*scale, color="black")
ax2.set_ylim(ax1.get_ylim())

imgMerge = ax3.imshow(Image.open(frames[0]), cmap = "copper", aspect = "auto")
arrowsMerge = ax3.quiver(np.arange(0, vectors.shape[1], step), np.flipud(np.arange(vectors.shape[0]-1, -1, -step)), 
                      vectors[::step, ::step, 0]*scale, vectors[::step, ::step, 1]*-1*scale, color="c")

x = (angles+np.pi) % (2*np.pi) - np.pi  #convert radians to degrees
n, bins = np.histogram(x, bins=32)                # ndarray bin data and record counts
widths = np.diff(bins)                          # ndarray; width of each bin
area = n / x.size                               # Area to assign each bin
radius = (area/np.pi) ** .5                     # Calculate corresponding bin radius  
patches = ax4.bar(x = bins[:-1], height = radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=False, linewidth=1)   # Plot data on ax
ax4.set_theta_offset(0)                         # Set the direction of the zero angle
ax4.set_yticks([])

def update(val):
    w = int(winSlider.val)
    n = int(nSlider.val)
    s = sSlider.val
    f = int(startSlider.val)
    newVectors = calcFlow(frame1 = Image.open(frames[f]), frame2 = Image.open(frames[f+framesToCompare]), pyr = 0.5, lev = 3, win = w, it = 3, polN = n, polS = s, flag = 1)
    newMags, newAngs = cv.cartToPolar(newVectors[:,:,0], newVectors[:,:,1]*-1, angleInDegrees = False) #multiple by -1 to get the angles along the same axis as image
    newAngles = newAngs.flatten()
    arrows.set_UVC(newVectors[::step, ::step, 0], newVectors[::step, ::step, 1]*-1) 
    arrowsMerge.set_UVC(newVectors[::step, ::step, 0], newVectors[::step, ::step, 1]*-1) 
    img.set_data(Image.open(frames[f]))  
    imgMerge.set_data(Image.open(frames[f]))  

    newX = (newAngles+np.pi) % (2*np.pi) - np.pi  #convert radians to degrees
    newN, newBins = np.histogram(newX, bins=32)                # Bin data and record counts
    newArea = newN / newX.size                               # Area to assign each bin
    newRadius = (newArea/np.pi) ** .5                     # Calculate corresponding bin radius  
    for i in range(len(newRadius)):
        patches[i].set_height(newRadius[i])
    fig.canvas.draw_idle()      #re-draws the plot

winSlider.on_changed(update)   #calls update function if slider is changed
nSlider.on_changed(update)   #calls update function if slider is changed
sSlider.on_changed(update)   #calls update function if slider is changed
startSlider.on_changed(update)   #calls update function if slider is changed
plt.show()