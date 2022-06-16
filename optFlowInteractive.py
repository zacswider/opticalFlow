import sys
import cv2 as cv
import numpy as np
from tkinter import Tk
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter.filedialog import askopenfilename

'''***** Default Parameters *****'''
Tk().withdraw()                 
imagePath = askopenfilename()       # full path to 1-channel image
#imagePath = " "    # If you want to hard code an image path
imageStack=skio.imread(imagePath) 
# scale variable for displayed vector size; bigger value = smaller vector
scale = 1
# step size for vectors. Larger value = less vectors displayed
step = 4
# skip frames when comparing. Default is to compare consecutive frames (i.e., skip zero)
framesToSkip = 0
 # number of bins for the polar histogram
numBins = 64
# starting frame to compare, default is to start comparing with the first frame (index 0)
start = 0
# sets first image frame
firstFrame = imageStack[start]
# sets second image frame
secondFrame = imageStack[start+framesToSkip+1]
#full of equally radian values around a circle
useBins = np.array([i*(6.28/numBins) for i in range(numBins + 1)])   

'''***** Quality Control *****'''
if imageStack.ndim > 3:
    print('The image has greater than three dimensions. Please load a single channel time lapse.')
    sys.exit()

'''***** Flow and Hist Functions *****'''
def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      
    # https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), 
                                       next = np.array(frame2), 
                                       flow = None,
                                       pyr_scale = pyr, 
                                       levels = lev,
                                        winsize = win, 
                                        iterations = it,
                                        poly_n = polN, 
                                        poly_sigma = polS, 
                                        flags = flag)
    return(flow)

def calcVectors(flowArray):
    # converts flow to magnitude and angels; multiple vectors by -1 to get the angles along the same axis as image
    mags, angs = cv.cartToPolar(flowArray[:,:,0], flowArray[:,:,1]*(-1), angleInDegrees = False)
    # flatten the angles and magnitudes to calculate the histogram
    flatAngles = angs.flatten()                           
    flatMagnitudes = mags.flatten()
    # n is the counts and bins is ndarray of the equally spaced bins between (-pi, pi)       
    n, bins = np.histogram(flatAngles, bins=useBins, weights = flatMagnitudes)
    # ndarray; width of each bin
    widths = np.diff(bins)
    # set the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height
    radius = n
    return(bins, widths, radius)

# calculates flow with default values
flow = calcFlow(frame1 = firstFrame, 
                frame2 = secondFrame, 
                pyr =0.5, 
                lev = 3, 
                win = 20, 
                it = 3, 
                polN = 7, 
                polS = 1.5, 
                flag = 1) 
# calculates histogram bins, widths, and heights
bins, widths, radius = calcVectors(flow)                    

'''***** Plots and Sliders *****'''
fig = plt.figure()                            
ax1 = plt.subplot(1, 2, 1)                    
ax2 = plt.subplot(1, 2, 2, projection='polar')
fig.subplots_adjust(top = 0.8)        

# create slider axes
winAx = fig.add_axes([.185, 0.9, 0.25, 0.05])       # rectangle of size [x0, y0, width, height]
winValues = np.linspace(1,100,100)                  
nAx = fig.add_axes([.185, 0.825, 0.25, 0.05])       
nValues = np.linspace(1,9,5)                        
sAx = fig.add_axes([.6, 0.9, 0.25, 0.05])           
sValues = np.linspace(0.1,2.0,20)                   
startAx = fig.add_axes([.6, 0.825, 0.25, 0.05])     

# creates sliders
startValues = np.linspace(0 ,imageStack.shape[0], imageStack.shape[0], endpoint=False)  # sets starting frame values
winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=20, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')                                 # window size slider parameters
nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')                                        # poly_n size slider parameters
sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')                                  # poly_sigma size slider parameters
startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=imageStack.shape[0], valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')       # start frame slider parameters

# show the first frame
imgMerge = ax1.imshow(imageStack[0], cmap = "copper", aspect = "equal") 
ax1.set_xticks([])
ax1.set_yticks([])

# matplotlib.quiver.Quiver object overlayed on axes image
arrowsMerge = ax1.quiver(np.arange(0, flow.shape[1], step), 
                         np.flipud(np.arange(flow.shape[0]-1, -1, -step)),   
                         flow[::step, ::step, 0]*scale, flow[::step, ::step, 1]*-1*scale, color="c")

# Bar plot using histogram data
histBars = ax2.bar(x = bins[:-1], 
                   height = radius, 
                   zorder=1, align='edge', 
                   width=widths, edgecolor='C0', 
                   fill=True,
                    linewidth=1)

# set the direction of the zero angle
ax2.set_theta_zero_location("E")
ax2.set_yticks([])

'''***** Update on Sliders *****'''
def update(val):
    # pull values from the sliders    
    w = int(winSlider.val)                  
    n = int(nSlider.val)
    s = sSlider.val
    f = int(startSlider.val) 

    # update the flow
    flow = calcFlow(frame1 = imageStack[f], 
                    frame2 = imageStack[f+framesToSkip+1], 
                    pyr = 0.5, 
                    lev = 3, 
                    win = w, 
                    it = 3, 
                    polN = n, 
                    polS = s, 
                    flag = 1)   
    
    # update arrows with new vector coordinates
    arrowsMerge.set_UVC(flow[::step, ::step, 0]*scale, 
                        flow[::step, ::step, 1]*-1*scale)    

    # update images to display
    imgMerge.set_data(imageStack[f])        # updates image to display

    # recalculate and reset new bar heights
    b, wi, radius = calcVectors(flow)
    for i in range(len(radius)):
        histBars[i].set_height(radius[i])

    # re-scales polar histogram y-axis
    ax2.set_ylim(top = np.max(radius))
    fig.canvas.draw_idle()

# call update function if slider is changed
winSlider.on_changed(update)                
nSlider.on_changed(update)                  
sSlider.on_changed(update)                  
startSlider.on_changed(update)              
plt.show()