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
        self.folder_path.set('/Users/bementmbp/Desktop/lesliecrop.tif')
        self.window_size = tk.IntVar()
        self.window_size.set(17)
        self.polyN_size = tk.IntVar()
        self.polyN_size.set(3)
        self.polyS_size = tk.DoubleVar()
        self.polyS_size.set(2)
        self.frame_skip_num = tk.IntVar()
        self.frame_skip_num.set(15)
        self.vector_skip_num = tk.IntVar()
        self.vector_skip_num.set(8)
        self.gauss_sigma = tk.IntVar()
        self.gauss_sigma.set(0)

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')
        self.folder_path.set('/Users/bementmbp/Desktop/Scripts/opticalFlow/test_data')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # entry widget for window size step
        self.window_size_entry = ttk.Entry(self, textvariable = self.window_size, width = 3)
        self.window_size_entry.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        self.window_size_entry = ttk.Label(self, text = 'Window size')
        self.window_size_entry.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        # entry widget for polyN size
        self.polyN_size_entry = ttk.Entry(self, textvariable = self.polyN_size, width = 3)
        self.polyN_size_entry.grid(row = 2, column = 0, padx = 10, sticky = 'E')
        self.polyN_size_entry = ttk.Label(self, text = 'PolyN (px)')
        self.polyN_size_entry.grid(row = 2, column = 1, padx = 10, sticky = 'W')

        # entry widget for polyS size
        self.polyS_size_entry = ttk.Entry(self, textvariable = self.polyS_size, width = 3)
        self.polyS_size_entry.grid(row = 3, column = 0, padx = 10, sticky = 'E')
        self.polyS_size_entry = ttk.Label(self, text = 'PolyS (sigma)')
        self.polyS_size_entry.grid(row = 3, column = 1, padx = 10, sticky = 'W')

        # entry widget for frame skip
        self.frame_skip_num_entry = ttk.Entry(self, textvariable = self.frame_skip_num, width = 3)
        self.frame_skip_num_entry.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        self.frame_skip_num_entry = ttk.Label(self, text = 'Frame to skip')
        self.frame_skip_num_entry.grid(row = 4, column = 1, padx = 10, sticky = 'W')

        # entry widget for vector skip
        self.vector_skip_num_entry = ttk.Entry(self, textvariable = self.vector_skip_num, width = 3)
        self.vector_skip_num_entry.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        self.vector_skip_num_entry = ttk.Label(self, text = 'Vector density')
        self.vector_skip_num_entry.grid(row = 5, column = 1, padx = 10, sticky = 'W')

        # entry widget for gauss sigma
        self.gauss_sigma_entry = ttk.Entry(self, textvariable = self.gauss_sigma, width = 3)
        self.gauss_sigma_entry.grid(row = 6, column = 0, padx = 10, sticky = 'E')
        self.gauss_sigma_entry = ttk.Label(self, text = 'Gaussian blur sigma')
        self.gauss_sigma_entry.grid(row = 6, column = 1, padx = 10, sticky = 'W')
    
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

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.folder_path = self.folder_path.get()
        self.window_size = self.window_size.get()
        self.polyN_size = self.polyN_size.get()
        self.polyS_size = self.polyS_size.get()
        self.frame_skip_num = self.frame_skip_num.get()
        self.vector_skip_num = self.vector_skip_num.get()
        self.gauss_sigma = self.gauss_sigma.get()
        


        # destroy the widget
        self.destroy()
