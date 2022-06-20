from optflowmods.flowgui import FlowGUI
from optflowmods.flowprocessor import FlowProcessor
import os

def main():
    gui = FlowGUI()
    gui.mainloop()

    folder_path = gui.folder_path
    window_size = gui.window_size
    polyN_size = gui.polyN_size
    polyS_size = gui.polyS_size
    frames_to_skip = gui.frame_skip_num
    vectors_to_skip = gui.vector_skip_num
    gauss_sigma = gui.gauss_sigma

    ims = [im_name for im_name in os.listdir(folder_path) if im_name.endswith('.tif') and not im_name.startswith('.')]

    fp = FlowProcessor(os.path.join(folder_path, ims[0]), custom_step_size, blur_sigma)


if __name__ == '__main__':
    main()