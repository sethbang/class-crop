import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import requests
import os

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

directory = ""

counts = {}


# This code takes in an input directory of images and an output directory to store the cropped images. It also takes in a confidence level for the model to determine if the object is present or not. It then loops through every image in the input directory and uses the model to detect objects in the image. It then crops the image to only include the detected object and saves it to the appropriate directory.

def image_crops(input_directory, output_directory, confidence):

    # Loop through every file in the input directory
    for filename in os.listdir(input_directory):
        # Get the path to the current file
        curr_path = os.path.join(input_directory, filename)

        # Open the current file as an image
        image = Image.open(curr_path)

        # Pass the image to the model to detect objects
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Get the dimensions of the image
        target_sizes = torch.tensor([image.size[::-1]])

        # Post process the model outputs to get the detected objects
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence)[0]

        # Loop through the detected objects
        with Image.open(curr_path) as im:
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Round the coordinates of the detected object
                box = [round(i, 2) for i in box.tolist()]
                # Get the label of the detected object
                label_text = model.config.id2label[label.item()]
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

                # Create a directory for the label if it does not exist
                if not (os.path.exists(f"{output_directory}/{label_text}")):
                    os.mkdir(f"{output_directory}/{label_text}")
                counts[label_text] = counts.get(label_text, 0) + 1
                remote_region = im.crop(box)
                remote_region.save(
                    f"{output_directory}/{label_text}/{label_text}_{counts[label_text]}.jpg")


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
    image_crops(input_dir_entry.get(), output_dir_entry.get(), slider.get())
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
