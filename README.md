# opticalFlow

The repository contains two python scripts for calculating the optical flow of image sequences using the openCV dense optical flow (Farneback’s) method. The flow parameters can be identified on a representative dataset using `optFlowInteractive.py` and then applied to n files with n channels by using `calcbatchflow.py`

## optFlowInteractive.py
 - When executed, this script will first open a finder window requesting an image sequence to calculate flow from. This image sequence must be no more and no less than three dimensions (x, y, and t). The file `200102_BZ_crop.tif` located within `./test_data` is available as a convenient test dataset.
 - Once a suitable file is chosen, the script launches an interative Matplotlib window with several sliders that can be used to adjust the optical flow parameters.

![alt text](https://github.com/zacswider/README_Images/blob/main/flow_int.png)

#### Interactive sliders
  - The following descriptions, where applicable, were taken directly from the OpenCV docs. [See the docs here](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback "OpenCV docs") 
 1) Averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
 2) Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
 3) Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field.
 4) The start frame. This frame will be compared with `start + skip + 1` subsequent frames
 5) The number of frames to skip.
 6) The number of vectors to skip. Plotting a couple hundred thousand vectors can be very time consuming and difficult to see. Skipping several vectors will speed up the process of re-drawing the plots and make the vector field easier to visualize.
 7) The sigma for a pre-calculation gaussian blur. This is largely unnecessary.

#### Interactive plots
 8) The two frames being compared in green and magenta
 9) The computed flow represented in HSV colour space
 10) The computed flow representated as a polar histogram weighted by vector magnitudes
 11) The spatial distribution of flow magnitude. There is no directional information here, just information about which pixels are moving the "most".
 12) Otsu threshold of the magnitudes overlayed with a vector representation of the calculated optical flow. The idea here is that many pixels in a given image may not be moving for any given frame. By filtering out the regions with the most movement, we can better estimate the properties of the objects that are moving.
 13) The computed flow magnitudes in the whole image are specifically in the masked regions displayed as box and whisker plots. 

 
Once a suitable combination of parameters are identified using the interactive script, you can batch calculate optical flow properties using calcbatchflow.py

## calcbatchflow.py

When executed, this script will open a GUI window asking for several parameters to calculate optical flow:
![alt text](https://github.com/zacswider/README_Images/blob/main/flow_gui.png)

 1) This is the source directory for your analysis. Navigate to it using the "Select source directory button". This directory should have one or more time lapse datasets saved in standard standard tzcyx order. Files may have as many channels as you want, but the flow will be calculated from all channels using the same parameters. If the data are not max projected along the z-axis prior to analysis, they will be max projected by the processing script.
 2) Averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
 3) Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field.
 4) Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
 5) The number of frames to skip. The default is to compare consecutive frame.
 6) The number of vectors to skip. Skipping some vectors will speed up the process of re-drawing the plots and make the vector field easier to visualize.
 7) The sigma for a pre-calculation gaussian blur. This is largely unnecessary.
 8) Start the analysis.

### Output

A new folder will be created in the analysis directory specified in step `1` above named `opt_flow_summary`. A new text file named `log.txt` will also be generated specifying the parameters used for the flow analysis. Each file in the analysis directory that was succesfully processed will be represented by an individual folder within the `opt_flow_summary` directory.  If a file was not processed, its name and the reason for being skipped will be added to the log file. Within each individual folder there will be, for each channel, two text files specifying the bin center and counts of flow magnitude for every pair of frames in the analyzed file. `masked_hist_ch(num).txt` uses only the pixels above Otsu's threshold for each pair of frames to calculate the histogram. `total_hist_ch(num).txt` uses every pixel in each pair of frames to calculate the histogram. There will also be a summary figure named `ch(num)_summary.png`, an example is shown below:

![alt text](https://github.com/zacswider/README_Images/blob/main/flow_output.png)

1) Histogram of pixel movements (pixels/frame) using either _all_ pixels (total) or just those above Otsu's threshold for each pair of images (masked)
2) Randomly chosen frame from this channel in the time series
3) Flow magnitudes from the frame in `2` (green) with Otsu's threshold overlaid (magenta)
4) Boxplot of pixel movements (pixels/frame) using either _all_ pixels (total) or just those above Otsu's threshold for each pair of images (masked)
5) Mean optical flow for the entire sequence displayed as a vector field
6) Mean optical flow for the entire sequence represented in HSV colour space

Finally, a new .csv file named `summary.csv` will be written in the analysis directory summarizing the total or masked mean and standard deviation of the optical flow measurements for each file in the analysis directory. 

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

Please see the environment.yml file to make your own environment.

