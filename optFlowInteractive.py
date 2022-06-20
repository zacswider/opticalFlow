import sys
import cv2 as cv
import numpy as np
from tkinter import Tk
import skimage.io as skio
import matplotlib.pyplot as plt
from pylab import *
import colour
from matplotlib.widgets import Slider
from tkinter.filedialog import askopenfilename
import scipy.ndimage as nd 

'''***** Default Parameters *****'''
#Tk().withdraw()                 
#imagePath = askopenfilename(initialdir = './test_data')       # full path to 1-channel image
imagePath = "/Users/bementmbp/Desktop/Scripts/opticalFlow/test_data/200102_BZ_crop.tif"    # If you want to hard code an image path
#imagePath = "/Users/bementmbp/Desktop/leslie.tif"
imageStack=skio.imread(imagePath) 
# scale variable for displayed vector size; bigger value = smaller vector
scale = 1
# step size for vectors. Larger value = less vectors displayed
vect_step = 2
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

brightness = 1000 

'''***** Quality Control *****'''
if imageStack.ndim > 3:
    print('The image has greater than three dimensions. Please load a single channel time lapse.')
    sys.exit()

'''***** Flow and Hist Functions *****'''
def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):      
    # https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev = np.invert(np.array(frame1)), 
                                       next = np.invert(np.array(frame2)), 
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
    # n is the counts and bins is ndarray of the equally spaced bins between (-pi, pi)       
    n, bins = np.histogram(angs.ravel(), bins=useBins, weights = mags.ravel())
    # ndarray; width of each bin
    widths = np.diff(bins)
    # set the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height
    radius = n
    return(mags, bins, widths, radius)

def draw_hsv(flow, brightness = 1000):                         #openCV function for drawing flow fields, with my modifications
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
mags, bins, widths, radius = calcVectors(flow)                    

'''***** Plots and Sliders *****'''
fig = plt.figure()      
ax1 = plt.subplot(2, 3, 1)                    
ax2 = plt.subplot(2, 3, 2, projection='polar')
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)
fig.subplots_adjust(top=0.65, wspace=0.05, hspace=0.05)       

# create slider axes
winAx = fig.add_axes([.185, 0.9, 0.25, 0.05])       # rectangle of size [x0, y0, width, height]
winValues = np.linspace(1,100,100)                  
nAx = fig.add_axes([.185, 0.825, 0.25, 0.05])       
nValues = np.linspace(1,9,5)                        
sAx = fig.add_axes([.6, 0.9, 0.25, 0.05])           
sValues = np.linspace(0.1,2.0,20)                   
startAx = fig.add_axes([.6, 0.825, 0.25, 0.05])
skipValues = np.linspace(0,25,25)
skipAx = fig.add_axes([.185, 0.75, 0.25, 0.05])
vect_step_values = np.linspace(1,16,16)
vect_step_values_ax = fig.add_axes([.6, 0.75, 0.25, 0.05])

# creates sliders
startValues = np.linspace(0 ,imageStack.shape[0], imageStack.shape[0], endpoint=False)  # sets starting frame values
winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=20, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')                                 # window size slider parameters
nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')                                        # poly_n size slider parameters
sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')                                  # poly_sigma size slider parameters
startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=imageStack.shape[0], valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')       # start frame slider parameters
skipSlider = Slider(ax=skipAx, label='skip', valmin=0, valmax=25, valinit=0, valfmt=' %0.0f frames', valstep=skipValues, facecolor='#cc7000')                           
vect_step_slider = Slider(ax=vect_step_values_ax, label='vect', valmin=0, valmax=16, valinit=0, valfmt=' %0.0f skip', valstep=vect_step_values, facecolor='#cc7000')

# show the first frame

def get_rgb(frame1, frame2):
    def normalize(im):
        return((im - im.min()) / (im.max() - im.min()))
    
    merge = np.zeros((frame1.shape[0], frame1.shape[1], 3))
    merge[:,:,0] = normalize(frame1)
    merge[:,:,1] = normalize(frame2)
    merge[:,:,2] = normalize(frame1)

    return(merge)

imgMerge = ax1.imshow(get_rgb(firstFrame, secondFrame), aspect = "equal") 
mags_img = ax3.imshow(mags, aspect = "equal")
for ax in (ax1, ax3):
    ax.set_xticks([])
    ax.set_yticks([])

# matplotlib.quiver.Quiver object overlayed on axes image
arrowsMerge = ax1.quiver(np.arange(0, flow.shape[1], vect_step), 
                         np.flipud(np.arange(flow.shape[0]-1, -1, -vect_step)),   
                         flow[::vect_step, ::vect_step, 0]*scale, flow[::vect_step, ::vect_step, 1]*-1*scale, color="c")

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

box = ax4.boxplot(mags.ravel())
x, y = box['medians'][0].get_xydata()[1]
mymean = str(round(mags.mean(),2))
mystd = str(round(mags.std(), 2))
text = f'Mean: {mymean} \nStd: {mystd}'
ax4.annotate(text, xy=(x, y))
ax4.set_ylabel('Pixel shift / frame')

'''***** Update on Sliders *****'''
def update(val):
    # pull values from the sliders    
    w = int(winSlider.val)                  
    n = int(nSlider.val)
    s = sSlider.val
    f = int(startSlider.val)
    skip = int(skipSlider.val)
    vs = int(vect_step_slider.val)
    first = imageStack[f]
    second = imageStack[f+skip+1]

    # update the flow
    flow = calcFlow(frame1 = first, 
                    frame2 = second, 
                    pyr = 0.5, 
                    lev = 3, 
                    win = w, 
                    it = 3, 
                    polN = n, 
                    polS = s, 
                    flag = 1)   
    
  

    # recalculate and reset new bar heights
    m, b, wi, radius = calcVectors(flow)
    for i in range(len(radius)):
        histBars[i].set_height(radius[i])

    # update images to display
    ax1.cla()
    ax1.imshow(get_rgb(first, second))
    # matplotlib.quiver.Quiver object overlayed on axes image
    ax1.quiver(np.arange(0, flow.shape[1], vs), 
                         np.flipud(np.arange(flow.shape[0]-1, -1, -vs)),   
                         flow[::vs, ::vs, 0]*scale, flow[::vs, ::vs, 1]*-1*scale, color="c")


    ax3.cla()
    ax3.imshow(m, aspect = "equal")

    # update ax4 boxplot
    ax4.cla()
    m = m / (1 + skip)
    box = ax4.boxplot(m.ravel())
    x, y = box['medians'][0].get_xydata()[1]
    mymean = str(round(m.mean(),2))
    mystd = str(round(m.std(), 2))
    text = f'Mean: {mymean} \nStd: {mystd}'
    ax4.annotate(text, xy=(x, y))

    # re-scales polar histogram y-axis
    ax2.set_ylim(top = np.max(radius))
    fig.canvas.draw_idle()

# call update function if slider is changed
winSlider.on_changed(update)                
nSlider.on_changed(update)                  
sSlider.on_changed(update)                  
startSlider.on_changed(update) 
skipSlider.on_changed(update)     
vect_step_slider.on_changed(update)        
plt.show()