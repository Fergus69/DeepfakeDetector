import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def select_folder():
    print("button_2 clicked")
    file_path = filedialog.askopenfilename()
    if file_path:
        messagebox.showinfo("Video Loaded", "Video successfully loaded for analysis.")
    return file_path    