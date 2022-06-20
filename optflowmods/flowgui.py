import sys
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

class FlowGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        self.geometry("600x235")
        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)

        # define variable types for the different widget field
        self.folder_path = tk.StringVar()
        self.manual_step = tk.BooleanVar()
        self.manual_step_size = tk.IntVar()
        self.manual_step_size.set(0)
        self.blur = tk.BooleanVar()
        self.blur_sigma = tk.IntVar()
        self.blur_sigma.set(0)

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')
        self.folder_path.set('/Users/bementmbp/Desktop/Scripts/opticalFlow/test_data')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # check box to manually enter a frame step
        self.manual_step_checkbox = ttk.Checkbutton(self, variable = self.manual_step)
        self.manual_step_checkbox.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        self.manual_step_checkbox['command'] = self.activateCheck
        self.manual_step_checkbox_label = ttk.Label(self, text = 'Enter a custom step size')
        self.manual_step_checkbox_label.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        # entry widget for manual frame step
        self.manual_step_entry = ttk.Entry(self, textvariable = self.manual_step_size, width = 3)
        self.manual_step_entry.grid(row = 2, column = 0, padx = 10, sticky = 'E')
        self.manual_step_entry.configure(state = 'disabled')
        self.manual_step_entry_label = ttk.Label(self, text = 'Enter a custom step size')
        self.manual_step_entry_label.grid(row = 2, column = 1, padx = 10, sticky = 'W')

        # check box to blur the image
        self.blur_checkbox = ttk.Checkbutton(self, variable = self.blur)
        self.blur_checkbox.grid(row = 3, column = 0, padx = 10, sticky = 'E')
        self.blur_checkbox['command'] = self.activateCheck
        self.blur_checkbox_label = ttk.Label(self, text = 'Blur the image')
        self.blur_checkbox_label.grid(row = 3, column = 1, padx = 10, sticky = 'W')

        # entry widget for blur sigma
        self.blur_sigma_entry = ttk.Entry(self, textvariable = self.blur_sigma, width = 3)
        self.blur_sigma_entry.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        self.blur_sigma_entry.configure(state = 'disabled')
        self.blur_sigma_entry_label = ttk.Label(self, text = 'Gaussian sigma value')
        self.blur_sigma_entry_label.grid(row = 4, column = 1, padx = 10, sticky = 'W')



        # create start button
        self.start_button = ttk.Button(self, text = 'Start analysis')
        self.start_button['command'] = self.start_analysis
        self.start_button.grid(row = 9, column = 0, padx = 10, sticky = 'E')

        # create cancel button
        self.cancel_button = ttk.Button(self, text = 'Cancel')
        self.cancel_button['command'] = self.cancel_analysis
        self.cancel_button.grid(row = 9, column = 1, padx = 10, sticky = 'W')

    def get_folder_path(self):
        self.folder_path.set(askdirectory())


    def activateCheck(self):
        if self.manual_step.get() == True:
            self.manual_step_entry.configure(state = 'normal')
        if self.manual_step.get() == False:
            self.manual_step_entry.configure(state = 'disabled')
        if self.blur.get() == True:
            self.blur_sigma_entry.configure(state = 'normal')
        if self.blur.get() == False:
            self.blur_sigma_entry.configure(state = 'disabled')

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.folder_path = self.folder_path.get()
        self.manual_step = self.manual_step.get()
        self.manual_step_size = self.manual_step_size.get()
        self.blur = self.blur.get()
        self.blur_sigma = self.blur_sigma.get()


        # destroy the widget
        self.destroy()
