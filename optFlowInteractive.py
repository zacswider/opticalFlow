from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

frames = list(Path('/Users/bementmbp/Desktop/testImages/eb1_crop').iterdir())
frames.sort()
step = 8
framesToCompare = 1
firstFrame = Image.open(frames[0])
secondFrame = Image.open(frames[0+framesToCompare])

def calcFlow(frame1, frame2, pyr, lev, win, it, polN, polS, flag):
    flow = cv.calcOpticalFlowFarneback(prev = np.array(frame1), next = np.array(frame2), flow = None, 
                                    pyr_scale = pyr, levels = lev, winsize = win, iterations = it, 
                                    poly_n = polN, poly_sigma = polS, flags = flag)
    return(flow)

vectors = calcFlow(frame1 = firstFrame, frame2 = secondFrame, pyr =0.5, lev = 3, win = 20, it = 3, polN = 7, polS = 1.5, flag = 0)


fig, ax = plt.subplots(figsize = (9,4.5))
ax = [ax, ax.twinx()]
fig.subplots_adjust(right=.7)

arrows = ax[1].quiver(np.arange(0, vectors.shape[1], step), np.flipud(np.arange(vectors.shape[0]-1, -1, -step)), 
                      vectors[::step, ::step, 0], vectors[::step, ::step, 1]*-1, color="c")
img = ax[0].imshow(Image.open(frames[0]), cmap = "copper", aspect = "auto")
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())

winAx = fig.add_axes([.8, 0.7, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
winValues = np.linspace(1,100,100)
pyrAx = fig.add_axes([.8, 0.6, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
pyrValues = np.linspace(0.1,0.9,9)
levAx = fig.add_axes([.8, 0.5, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
levValues = np.array([1,2,3,4,5])
itAx = fig.add_axes([.8, 0.4, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
itValues = np.linspace(1,100,100)
nAx = fig.add_axes([.8, 0.3, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
nValues = np.linspace(1,9,5)
sAx = fig.add_axes([.8, 0.2, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
sValues = np.linspace(0.1,2.0,20)
startAx = fig.add_axes([.8, 0.1, 0.1, 0.05]) #rectangle of size [x0, y0, width, height]
startValues = np.linspace(0,len(frames),len(frames))

try:
    winSlider = Slider(ax=winAx, label='win', valmin=1, valmax=100, valinit=20, valfmt=' %0.0f Px', valstep=winValues, facecolor='#cc7000')
except ValueError:
    winSlider = Slider(ax=winAx, label='test', valmin=5, valmax=50, valinit=20, valfmt=' %0.0f Px', valstep=winValues.all(), facecolor='#cc7000')
try:
    pyrSlider = Slider(ax=pyrAx, label='pyr', valmin=0.1, valmax=0.9, valinit=0.5, valfmt=' %1.1f Scale', valstep=pyrValues, facecolor='#cc7000')
except ValueError:
    pyrSlider = Slider(ax=pyrAx, label='pyr', valmin=0.1, valmax=0.9, valinit=0.5, valfmt=' %1.1f Scale', valstep=pyrValues.all(), facecolor='#cc7000')
try:
    levSlider = Slider(ax=levAx, label='lev', valmin=1, valmax=5, valinit=3, valfmt=' %0.0f Levels', valstep=levValues, facecolor='#cc7000')
except ValueError:
    levSlider = Slider(ax=levAx, label='lev', valmin=1, valmax=5, valinit=3, valfmt=' %0.0f Levels', valstep=levValues.all(), facecolor='#cc7000')
try:
    itSlider = Slider(ax=itAx, label='its', valmin=1, valmax=100, valinit=3, valfmt=' %0.0f Iters', valstep=itValues, facecolor='#cc7000')
except ValueError:
    itSlider = Slider(ax=itAx, label='its', valmin=1, valmax=100, valinit=3, valfmt=' %0.0f Iters', valstep=itValues.all(), facecolor='#cc7000')
try:
    nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues, facecolor='#cc7000')
except ValueError:
    nSlider = Slider(ax=nAx, label='polyN', valmin=1, valmax=9, valinit=7, valfmt=' %0.0f Px', valstep=nValues.all(), facecolor='#cc7000')
try:
    sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues, facecolor='#cc7000')
except ValueError:
    sSlider = Slider(ax=sAx, label='polyS', valmin=0.1, valmax=2.0, valinit=1.5, valfmt=' %1.1f Px', valstep=sValues.all(), facecolor='#cc7000')
try:
    startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=len(frames), valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')
except ValueError:
    startSlider = Slider(ax=startAx, label='start', valmin=0, valmax=len(frames), valinit=0, valfmt='Frame %0.0f ', valstep=startValues, facecolor='#cc7000')

def update(val):
    w = int(winSlider.val)
    print("win = ", w)
    p = pyrSlider.val
    print("pyramid = ", p)
    l = int(levSlider.val)
    print("levels = ", l)
    i = int(itSlider.val)
    print("iterations = ", i)
    n = int(nSlider.val)
    print("polyN = ", n)
    s = sSlider.val
    print("polyS = ", s)
    f = int(startSlider.val)
    print("frame = ", f)
    newVectors = calcFlow(frame1 = Image.open(frames[f]), frame2 = Image.open(frames[f+framesToCompare]), pyr = p, lev = l, win = w, it = i, polN = n, polS = s, flag = 1)
    arrows.set_UVC(newVectors[::step, ::step, 0], newVectors[::step, ::step, 1]*-1) 
    img.set_data(Image.open(frames[f]))  
    fig.canvas.draw_idle()      #re-draws the plot

winSlider.on_changed(update)   #calls update function if slider is changed
pyrSlider.on_changed(update)   #calls update function if slider is changed
levSlider.on_changed(update)   #calls update function if slider is changed
itSlider.on_changed(update)   #calls update function if slider is changed
nSlider.on_changed(update)   #calls update function if slider is changed
sSlider.on_changed(update)   #calls update function if slider is changed
startSlider.on_changed(update)   #calls update function if slider is changed

plt.show()