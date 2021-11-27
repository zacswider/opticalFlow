# opticalFlow
Python script for visualizing optical flow from image sequences

optFlowInteractive.py uses a matplotlib slider interface to optimize window size, poly_n, and poly_sigma values for the openCV Lucas-Kanade dense optical flow method. This script currently accepts the full path to a one-channel time lapse, which is read in and analyzed as a numpy ndarray. The matplotlib window shows an image frame overlayed with the vector field on the left, and a weighted histogram of the vectors on the right (example below).

![alt text](https://github.com/zacswider/opticalFlow/blob/inProgress/exampleInterface.jpg)

Dependencies:
- numpy
- opencv
- matplotlib
- scikit-image



