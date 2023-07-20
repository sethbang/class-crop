import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import requests
import os


def select_input_dir():
    folder_selected = filedialog.askdirectory()
    input_dir_entry.delete(0, tk.END)
    input_dir_entry.insert(0, folder_selected)


def select_output_dir():
    folder_selected = filedialog.askdirectory()
    output_dir_entry.delete(0, tk.END)
    output_dir_entry.insert(0, folder_selected)


def submit():
    print(f"Input Directory: {input_dir_entry.get()}")
    print(f"Output Directory: {output_dir_entry.get()}")
    print(f"Confidence: {slider.get()}")
    root.destroy()


root = tk.Tk()

tk.Button(root, text="Select Input Folder",
          command=select_input_dir).grid(row=0, column=0)
input_dir_entry = tk.Entry(root, width=50)
input_dir_entry.grid(row=0, column=1)

tk.Button(root, text="Select Output Folder",
          command=select_output_dir).grid(row=1, column=0)
output_dir_entry = tk.Entry(root, width=50)
output_dir_entry.grid(row=1, column=1)

tk.Label(root, text="Confidence").grid(row=2, column=0)
slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
slider.set(0.9)
slider.grid(row=3, column=0)

tk.Button(root, text="Submit", command=submit).grid(
    row=4, column=0, columnspan=2)

root.mainloop()
