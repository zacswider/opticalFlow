import sys
import cv2 as cv
import numpy as np
from tkinter import Tk
import skimage.io as skio
import matplotlib.pyplot as plt
import colour
from matplotlib.widgets import Slider
from tkinter.filedialog import askopenfilename
import scipy.ndimage as nd
from skimage.filters import threshold_otsu
import time

'''***** Default Parameters *****'''
#Tk().withdraw()                 
#imagePath = askopenfilename(initialdir = './test_data')       # full path to 1-channel image
imagePath = "/Users/bementmbp/Desktop/Scripts/opticalFlow/test_data/200102_BZ_crop.tif"    # If you want to hard code an image path
imagePath = "/Users/bementmbp/Desktop/leslie.tif"
imageStack=skio.imread(imagePath) 

scale = 1                                                               # scale variable for displayed vector size; bigger value = smaller vector
vect_step = 2                                                           # step size for vectors. Larger value = less vectors displayed
framesToSkip = 0                                                        # skip frames when comparing. Default is to compare consecutive frames (i.e., skip zero)
numBins = 64                                                            # number of bins for the polar histogram
start = 0                                                               # starting frame to compare
first_frame = imageStack[start][20:-20,20:-20]                          # sets first image frame
second_frame = imageStack[start+framesToSkip+1][20:-20,20:-20]          # sets second image frame
useBins = np.array([i*(6.28/numBins) for i in range(numBins + 1)])      # full of equally radian values around a circle


'''***** Quality Control *****'''
if imageStack.ndim > 3:
    print('The image has greater than three dimensions. Please load a single channel time lapse.')
    sys.exit()

'''***** Functions *****'''
def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):
    '''
    Calculates optical flow between two frames using the Farneback algorithm.
    docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    Returns a ndarray of the flow vectors
    '''
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
    '''
    Converts flow vectors to magnitude and angles. 
    Returns magnitudes as well as histogram bin properties
    '''
    # converts flow to magnitude and angels; multiple vectors by -1 to get the angles along the same axis as image
    mags, angs = cv.cartToPolar(flowArray[:,:,0], flowArray[:,:,1]*(-1), angleInDegrees = False)
    # n is the counts and bins is ndarray of the equally spaced bins between (-pi, pi)       
    n, bins = np.histogram(angs.ravel(), bins=useBins, weights = mags.ravel())
    # ndarray; width of each bin
    widths = np.diff(bins)
    # set the histogram radius equal to counts; could modify this to set bar *area* proportional to counts instead of height
    radius = n
    return(mags, bins, widths, radius)

def draw_hsv(flow, corr):
    '''
    OpenCV function for converting flow to HSV with my modifications.
    Corr is a correction factor for the brightness of the displayed values
    '''
    brightness = 1000 / (corr + 1)
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

def colour_wheel(samples=1024, clip_circle=True, method='Colour'): 
    '''
    Returns a ndarray of RGB values for a colour wheel.
    Modified from stackoverflow.com/a/62544063/4812591
    '''
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

def merge_frames(frame1, frame2):
    '''
    Normalizes the intensity of two 2D ndarrays (grayscale images) 
    and returns a 3D ndarray of the two frames merged into a single 
    RGB image with frame1 displayed magenta and frame2 displayed green.
    '''
    def normalize(im):
        return((im - im.min()) / (im.max() - im.min()))
    
    merge = np.zeros((frame1.shape[0], frame1.shape[1], 3))
    merge[:,:,0] = normalize(frame1)
    merge[:,:,1] = normalize(frame2)
    merge[:,:,2] = normalize(frame1)

    return(merge)

def plot_boxes(ax, data: list, labels: list):
    '''
    Accepts an xis object, list of arrays of plot, and a list of labels
    to assign to the plotted arrays. Calculated the mean and std of each
    plotted array and annotates the values next to each plot.
    '''
    boxes = ax.boxplot(data, labels = labels)
    for i in range(3):
        x, y = boxes['medians'][i].get_xydata()[1]
        data_mean = str(round(np.mean(data[i]), 2))
        data_std = str(round(np.std(data[i]), 2))
        text = f'Mean: {data_mean}\nStd: {data_std}'
        ax.text(x, y, text, fontsize = 'xx-small', color = '#cc7000')                  

'''***** Plots and Sliders *****'''
fig = plt.figure(figsize=(12,8))      
ax1 = plt.subplot(2, 3, 1)                    
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3, projection='polar')
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)
fig.subplots_adjust(top=0.78, wspace=0.1, hspace=0.2)

# create slider axes and value ranges
winAx = fig.add_axes([.185, 0.9, 0.25, 0.025])       # rectangle of size [x0, y0, width, height]
winValues = np.linspace(1,100,100)                  
nAx = fig.add_axes([.185, 0.875, 0.25, 0.025])       
nValues = np.linspace(1,9,5)                        
sAx = fig.add_axes([.6, 0.9, 0.25, 0.025])           
sValues = np.linspace(0.1,2.0,20)                   
startAx = fig.add_axes([.6, 0.875, 0.25, 0.025])
skipValues = np.linspace(0,25,25)
skipAx = fig.add_axes([.185, 0.85, 0.25, 0.025])
vect_step_values = np.linspace(1,16,16)
vect_step_values_ax = fig.add_axes([.6, 0.85, 0.25, 0.025])
gauss_step_values = np.linspace(0,16,16)
gauss_step_values_ax = fig.add_axes([.185, 0.825, 0.25, 0.025])

# creates sliders
startValues = np.linspace(0 ,imageStack.shape[0], imageStack.shape[0], endpoint=False)  # sets starting frame values
winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=20, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')                                 # window size slider parameters
nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')                                        # poly_n size slider parameters
sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')                                  # poly_sigma size slider parameters
startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=imageStack.shape[0], valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')       # start frame slider parameters
skipSlider = Slider(ax=skipAx, label='skip', valmin=0, valmax=25, valinit=0, valfmt=' %0.0f frames', valstep=skipValues, facecolor='#cc7000')                           
vect_step_slider = Slider(ax=vect_step_values_ax, label='vect', valmin=1, valmax=16, valinit=2, valfmt=' %0.0f skip', valstep=vect_step_values, facecolor='#cc7000')
gauss_step_slider = Slider(ax=gauss_step_values_ax, label='gauss', valmin=0, valmax=16, valinit=0, valfmt=' %0.0f sigma', valstep=gauss_step_values, facecolor='#cc7000')

''' ***** Plotting ***** '''
# calculates flow with default values
flow = calcFlow(frame1 = np.invert(first_frame), 
                frame2 = np.invert(second_frame), 
                pyr =0.5, 
                lev = 3, 
                win = 20, 
                it = 3, 
                polN = 7, 
                polS = 1.5, 
                flag = 1) 
# calculates histogram bins, widths, and heights
mags, bins, widths, radius = calcVectors(flow)  

# show images
ax1.imshow(merge_frames(first_frame, second_frame), aspect = "equal") 
ax1.set_xlabel('Magenta = frame #1\nGreen = frame #2')
ax2.imshow(draw_hsv(flow, corr = framesToSkip))
ax2.set_xlabel('Flow: HSV')
ax4.imshow(mags, aspect = "equal")
ax4.set_xlabel('Flow magnitude')

# get and set color wheel
pos = ax2.get_position()
newax = fig.add_axes([pos.x0, pos.y0, 0.05, 0.05], anchor='NW')#, zorder=-1)
wheel = colour_wheel(samples=64)
rotatedWheel = nd.rotate(wheel, 180)
newax.imshow(rotatedWheel) #ROTATE 180
newax.axis('off')

# matplotlib.quiver.Quiver object overlayed on axes image
ax5.quiver(np.arange(0, flow.shape[1], vect_step), 
                    np.flipud(np.arange(flow.shape[0]-1, -1, -vect_step)),   
                    flow[::vect_step, ::vect_step, 0]*scale, 
                    flow[::vect_step, ::vect_step, 1]*-1*scale, 
                    color="c")

# Bar plot using histogram data
histBars = ax3.bar(x = bins[:-1], 
                   height = radius, 
                   zorder=1, align='edge', 
                   width=widths, edgecolor='C0', 
                   fill=True,
                   linewidth=1)
ax3.set_theta_zero_location("E")
ax3.set_yticks([])

# generate otsu threshold of flow magnitude
mags_thresh = threshold_otsu(mags)
mags_mask = mags > mags_thresh
filtered_mags = mags[mags_mask]

# generate otsu threshold of raw frame 1
frame1_thresh = threshold_otsu(np.maximum(first_frame, second_frame))
frame1_mask = np.maximum(first_frame, second_frame) > frame1_thresh
filtered_frame1 = mags[frame1_mask]

masks_merged = np.zeros((mags.shape[0], mags.shape[1], 3))
masks_merged[:,:,0] = mags_mask
masks_merged[:,:,1] = frame1_mask
masks_merged[:,:,2] = mags_mask
ax5.imshow(masks_merged, aspect = "equal")
ax5.set_xlabel('Magenta = Otsu threshhold of mags\nGreen = Otsu threshold of Frame1')
    
plot_boxes(ax6, 
           [mags.ravel(), filtered_mags.ravel(), filtered_frame1.ravel()], 
           ['All pixels', 'Thesh mags', 'Thresh waves'])

ax6.set_ylabel('Pixel shift / frame')
ax6.tick_params(axis = 'x', labelrotation = 45)

for ax in (ax1, ax2, ax4, ax5):
    ax.set_xticks([])
    ax.set_yticks([])

'''***** Update on Sliders *****'''
def update(val):
    print('recalculating...')
    start = time.time()
    # pull values from the sliders    
    w = int(winSlider.val)                  
    n = int(nSlider.val)
    s = sSlider.val
    f = int(startSlider.val)
    skip = int(skipSlider.val)
    vs = int(vect_step_slider.val)
    sig = int(gauss_step_slider.val)

    # identify the frames to use for flow calculation
    first = nd.gaussian_filter(imageStack[f][w:-w,w:-w], sigma = sig)
    second = nd.gaussian_filter(imageStack[f+skip+1][w:-w,w:-w], sigma = sig)

    # calculate the optical flow
    flow = calcFlow(frame1 = np.invert(first), 
                    frame2 = np.invert(second), 
                    pyr = 0.5, 
                    lev = 3, 
                    win = w, 
                    it = 3, 
                    polN = n, 
                    polS = s, 
                    flag = 1)   

    # update ax1
    ax1.cla()
    ax1.imshow(merge_frames(first, second))
    ax1.set_xlabel('Magenta = frame #1\nGreen = frame #2')

    # update ax2
    ax2.cla()
    ax2.imshow(draw_hsv(flow, corr = skip))
    ax2.set_xlabel('Flow: HSV')

    # update ax3
    m, b, wi, radius = calcVectors(flow)
    for i in range(len(radius)):
        histBars[i].set_height(radius[i])
    ax3.set_ylim(top = np.max(radius))

    # update ax4
    ax4.cla()
    ax4.imshow(m, aspect = "equal")
    ax4.set_xlabel('Flow magnitude')

    # update ax5
    ax5.cla()

    mags_thresh = threshold_otsu(m)
    mags_mask = m > mags_thresh
    filtered_mags = m[mags_mask]

    frame1_thresh = threshold_otsu(np.maximum(first, second))
    frame1_mask = np.maximum(first, second) > frame1_thresh
    filtered_frame1 = m[frame1_mask]

    masks_merged = np.zeros((m.shape[0], m.shape[1], 3))
    masks_merged[:,:,0] = mags_mask
    masks_merged[:,:,1] = frame1_mask
    masks_merged[:,:,2] = mags_mask
    ax5.imshow(masks_merged, aspect = "equal")
    ax5.quiver(np.arange(0, flow.shape[1], vs), 
                         np.flipud(np.arange(flow.shape[0]-1, -1, -vs)),   
                         flow[::vs, ::vs, 0]*scale, flow[::vs, ::vs, 1]*-1*scale, color="c")
    ax5.set_xlabel('Magenta = Otsu threshhold of mags\nGreen = Otsu threshold of Frame1')

    # update ax6
    ax6.cla()
    m = m / (1 + skip)
    filtered_mags = filtered_mags / (1 + skip)
    filtered_frame1 = filtered_frame1 / (1 + skip)
    plot_boxes(ax6, 
              [mags.ravel(), filtered_mags.ravel(), filtered_frame1.ravel()], 
              ['All pixels', 'Thesh mags', 'Thresh waves'])
    ax6.set_ylabel('Pixel shift / frame')
    ax6.tick_params(axis = 'x', labelrotation = 45)

    for ax in (ax1, ax2, ax4, ax5):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.draw_idle()
    end = time.time()
    print(f'Recalculation finished in {round(end - start, 3)} seconds')

# call update function if slider is changed
winSlider.on_changed(update)                
nSlider.on_changed(update)                  
sSlider.on_changed(update)                  
startSlider.on_changed(update) 
skipSlider.on_changed(update)     
vect_step_slider.on_changed(update)
gauss_step_slider.on_changed(update)  

plt.show()