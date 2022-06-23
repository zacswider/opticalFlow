# opticalFlow

The repository contains two python scripts for calculating the optical flow of image sequences using the openCV dense optical flow (Farneback’s) method. The flow parameters can be identified on a representative dataset using `optFlowInteractive.py` and then applied to n files with n channels by using `calcbatchflow.py`

### optFlowInteractive.py
 - When executed, this script will first open a finder window requesting an image sequence to calculate flow from. This image sequence must be no more and no less than three dimensions (x, y, and t). The file `200102_BZ_crop.tif` located within `./test_data` is available as a convenient test dataset.
 - Once a suitable file is chosen, the script launches an interative Matplotlib window with several sliders that can be used to adjust the optical flow parameters.

![alt text](https://github.com/zacswider/README_Images/blob/main/flowgui.png)

#### Interactive plots
 1) The two frames being compared in green and magenta
 2) The computed flow represented in HSV colour space
 3) The computed flow representated as a polar histogram weighted by vector magnitudes
 4) The spatial distribution of flow magnitude. There is no directional information here, just information about which pixels are moving the "most".
 5) Otsu threshold of the magnitudes overlayed with a vector representation of the calculated optical flow. The idea here is that many pixels in a given image may not be moving for any given frame. By filtering out the regions with the most movement, we can better estimate the properties of the objects that are moving.
 6) The computed flow magnitudes in the whole image are specifically in the masked regions displayed as box and whisker plots. 
#### Interactive sliders
  - The following descriptions, where applicable, were taken directly from the OpenCV docs. [See the docs here](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback "OpenCV docs") 

 a) Averaging window size;larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
 
 b) Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field.
 
 c) The number of frames to skip.
 
 d) The sigma for a pre-calculation gaussian blur. This is largely unnecessary.
 
 e) Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
 
 f) The start frame. This frame will be compared with `start + skip + 1` subsequent frames
 
 g) The number of vectors to skip. Plotting a couple hundred thousand vectors can be very time consuming and difficult to see. Skipping several vectors will speed up the process of re-drawing the plots and make the vector field easier to visualize.
 
Once a suitable combination of parameters are identified using the interactive script, you can batch calculate optical flow properties using calcbatchflow.py

### calcbatchflow.py

BRB wrapping this part up still



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

