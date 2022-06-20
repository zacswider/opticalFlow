from optflowmods.flowgui import FlowGUI
from optflowmods.flowprocessor import FlowProcessor
import os

def main():
    gui = FlowGUI()
    gui.mainloop()

    folder_path = gui.folder_path
    custom_step = gui.manual_step
    blur = gui.blur
    custom_step_size = gui.manual_step_size
    blur_sigma = gui.blur_sigma

    ims = [im_name for im_name in os.listdir(folder_path) if im_name.endswith('.tif') and not im_name.startswith('.')]

    fp = FlowProcessor(os.path.join(folder_path, ims[0]), custom_step_size, blur_sigma)


if __name__ == '__main__':
    main()