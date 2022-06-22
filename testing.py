from optflowmods.flowprocessor import FlowProcessor
import numpy as np
import matplotlib.pyplot as plt

im_path = '/Users/bementmbp/Desktop/lesliecrop.tif'
im_path = '/Users/bementmbp/Desktop/Scripts/opticalFlow/test_data/200102_BZ_crop.tif'
window_size = 15
polyN_size = 3
polyS_size = 2
frames_to_skip = 1
vectors_to_skip = 2
gauss_sigma = 0

fp = FlowProcessor( image_path=im_path,
                    win_size = window_size,
                    polyN = polyN_size,
                    polyS = polyS_size,
                    frame_skip = frames_to_skip,
                    vect_skip = vectors_to_skip,
                    gauss_sigma = gauss_sigma  )

flow_array = fp.calc_mean_flow()
mags_array, masked_mags_array = fp.calc_regional_flow()
mean_mags_array = np.mean(mags_array, axis = (2,3))
plots = fp.plot_summary()

print(fp.image.shape)


def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

for j in range(fp.num_channels):
    show_figure(plots[f'Ch{j + 1}'])

plt.show()