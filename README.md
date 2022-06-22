# opticalFlow
Python scripts for 
1) Interactively modulating the variables for openCV dense optical flow (Farneback’s) method and visualizing optical flow.
2) Batch calculating mean optical flow using parameters determined in part 1.

optFlowInteractive.py uses a matplotlib slider interface to optimize window size, poly_n, and poly_sigma values for the openCV dense optical flow (Farneback’s) method. This script currently accepts one-channel time lapse images:

![alt text](https://github.com/zacswider/README_Images/blob/main/flowgui.png)

Dependencies:
  - numpy
  - opencv
  - tqdm
  - tk
  - pandas
  - matplotlib
  - tifffile
  - scipy
  - scikit-image
  - colour-science

Please see the environment .yml file to make your own environment.

