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
        
        #sets number of columns in the main window
        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)

        # define variable types for the different widget field
        self.folder_path = tk.StringVar()
        self.registration = tk.BooleanVar()
        self.framestep = tk.IntVar()

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')
        # make a default path
        self.folder_path.set('../test_data/200102_BZ_crop.tif')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # data registration widget
        self.registration_checkbox = ttk.Checkbutton(self, variable = self.registration)
        self.registration_checkbox.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        # create box size label text
        self.registration_checkbox = ttk.Label(self, text = 'register data for drift?')
        self.registration_checkbox.grid(row = 1, column = 1, padx = 10, sticky = 'W')




        
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
        self.registration = self.registration.get()


        # destroy the widget
        self.destroy()
