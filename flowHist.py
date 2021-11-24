import numpy as np
from pathlib import Path
from PIL import Image
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from skimage.feature import hog
from matplotlib import pyplot as plt
import cv2 as cv

first = cv.imread("/Users/bementmbp/Desktop/testImages/bzOptFlow/allFrames0000.tif")
second = cv.imread("/Users/bementmbp/Desktop/testImages/bzOptFlow/allFrames0002.tif")

frames = list(Path('/Users/bementmbp/Desktop/testImages/bzOptFlow').iterdir())
frames.sort()
step = 4
framesToSkip = 1
firstFrame = Image.open(frames[0])
secondFrame = Image.open(frames[0+1+framesToSkip])

def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)

vectors = calcFlow(frame1 = firstFrame, frame2 = secondFrame, pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 0)
mags, angs = cv.cartToPolar(vectors[:,:,0], vectors[:,:,1]*-1, angleInDegrees = False) #multiple by -1 to get the angles along the same axis as image
angles = angs.flatten()
magnitudes = mags.flatten()

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Function written by jwalton and described here: https://stackoverflow.com/a/55067613/4812591
    Parameters:
    ax : matplotlib.axes._subplots.PolarAxesSubplot; axis instance created with subplot_kw=dict(projection='polar').
    x : array; Angles to plot, expected in units of radians.
    bins : int, optional; Defines the number of equal-width bins in the range. The default is 16.
    density : bool, optional; If True plot frequency proportional to area. If False plot frequency proportional to radius. The default is True.
    offset : float, optional; Sets the offset for the location of the 0 direction in units of radians. The default is 0.
    gaps : bool, optional; Whether to allow gaps between bins. When gaps = False the bins are forced to partition the entire [-pi, pi] range. The default is True.
    Returns: 
    n : array or list of arrays. The number of values in each bin.
    bins : array. The edges of the bins.
    patches : `.BarContainer` or list of a single `.Polygon` Container of individual artists used to create the histogram or list of such containers if there are multiple input datasets.
    """
    
    x = (x+np.pi) % (2*np.pi) - np.pi                   # Wrap angles to [-pi, pi)
    if not gaps:                                        # Force bins to partition entire circle
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    n, bins = np.histogram(x, bins=bins)                # Bin data and record counts
    widths = np.diff(bins)                              # Compute width of each bin

    if density:                                         # By default plot frequency proportional to area
        area = n / x.size                               # Area to assign each bin
        radius = (area/np.pi) ** .5                     # Calculate corresponding bin radius  
    else:                                               # Otherwise plot frequency proportional to radius
        radius = n

    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths, edgecolor='C0', fill=False, linewidth=1)   # Plot data on ax
    ax.set_theta_offset(offset)                         # Set the direction of the zero angle
    
    if density:                                         # Remove ylabels for area plots (they are mostly obstructive)
        ax.set_yticks([])

    return n, bins, patches

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
circular_hist(ax, angles)
plt.show()