
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8

#gui
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, messagebox
import sys
sys.path.insert(0, "./Utilities")
import Utilities.functions as functions
import shutil
from PIL import Image



#alg
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import os
import cv2


#functions
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path('./assets/frame0')


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def select_file():
    print("button_2 clicked")
    suffix=None
    file_path = filedialog.askopenfilename()
    if file_path:
        suffix=Path(file_path).suffix
        file_extension = Path(file_path).suffix
        if file_extension == '.png':
            messagebox.showinfo("Photo Loaded", "Photo successfully loaded for analysis.")
            return file_path, suffix
        elif file_extension == '.mp4':
            messagebox.showinfo("Video Loaded", "Video successfully loaded for analysis.")
            return file_path, suffix
        else:
             messagebox.showinfo("Wrong type", "This application only supports png and mp4 files.")    
    return None
    

def save_all_frames_of_video(video_path, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Capture the video from the given path
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read
        
        # Construct the output path with a unique filename for each frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
        
        # Save the current frame as a PNG file
        cv2.imwrite(frame_filename, frame)
        print(f"Frame {frame_count} saved to {frame_filename}")
        
        frame_count += 1
    
    # When everything done, release the video capture object
    cap.release()
    print("All frames have been saved.")
    
def resize_images_in_same_folder(folder_path, size=(256, 256)):
    """
    Resize all images in the specified folder to the given size and save them back to the same folder.

    Parameters:
    - folder_path: Path to the folder containing the images to resize.
    - size: A tuple specifying the new size as (width, height).
    """
    folder_path = Path(folder_path)

    for file in folder_path.iterdir():
        if file.is_file() and file.suffix in ['.jpg', '.jpeg', '.png']:
            try:
                img = Image.open(file)
                img = img.resize(size, Image.ANTIALIAS)
                img.save(file)  # Save the image back to the same file
                print(f"Resized and saved {file.name}")
            except Exception as e:
                print(f"Failed to resize {file.name}. Reason: {e}")


def delete_image():
    canvas.delete(image_2)  # Clear the canvas

def create_image(path):
    poza = PhotoImage(file=path)
    image_2 = canvas.create_image(
    581.0,
    377.0,
    image=poza
    )
    image_2.image=poza

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

predict_folder = './build/assets/predict'
def button_2_func():
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
    else:
        # Clear the contents of the predict folder
        clear_directory(predict_folder)
    fisier,suffix=select_file()
    if fisier:
        if suffix == '.png':
            delete_image()
            create_image(fisier)
        elif suffix == '.mp4':
            save_all_frames_of_video(fisier, './build/assets/predict/')



# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)
def alg():
    image_dimensions = {'height':256, 'width':256, 'channels':3}

    # Create a Classifier class

    class Classifier:
        def __init__(self):
            self.model = 0

        def predict(self, x):
            return self.model.predict(x)

        def fit(self, x, y):
            return self.model.train_on_batch(x, y)

        def get_accuracy(self, x, y):
            return self.model.test_on_batch(x, y)

        def load(self, path):
            self.model.load_weights(path)

    class Meso4(Classifier):
        def __init__(self, learning_rate = 0.001):
            self.model = self.init_model()
            optimizer = Adam(learning_rate)
            self.model.compile(optimizer = optimizer,
                               loss = 'mean_squared_error',
                               metrics = ['accuracy'])

        def init_model(self): 
            x = Input(shape = (image_dimensions['height'],
                               image_dimensions['width'],
                               image_dimensions['channels']))

            x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
            x1 = BatchNormalization()(x1)
            x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

            x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
            x2 = BatchNormalization()(x2)
            x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

            x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
            x3 = BatchNormalization()(x3)
            x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

            x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
            x4 = BatchNormalization()(x4)
            x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

            y = Flatten()(x4)
            y = Dropout(0.5)(y)
            y = Dense(16)(y)
            y = LeakyReLU(negative_slope=0.1)(y)
            y = Dropout(0.5)(y)
            y = Dense(1, activation = 'sigmoid')(y)

            return tf.keras.models.Model(inputs = x, outputs = y)

    # Instantiate a MesoNet model with pretrained weights
    print("Current working directory: ", os.getcwd())
    meso = Meso4()
    meso.load('./algoritm/weights/Meso4_DF.h5')

    # Prepare image data

    # Rescaling pixel values (between 1 and 255) to a range between 0 and 1
    dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Instantiating generator to feed images through the network
    generator = dataGenerator.flow_from_directory(
        './algoritm/data/',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Checking class assignment
    generator.class_indices

    # Rendering image X with label y for MesoNet
    X, y = next(generator)

    # Evaluating prediction
    print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
    print(f"Actual label: {int(y[0])}")
    print(f"\nCorrect prediction: {round(meso.predict(X)[0][0])==y[0]}")

    # Showing image
    plt.imshow(np.squeeze(X))



window = Tk()

window.geometry("1161x946")
window.configure(bg = "#FFFFFF")


image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 946,
    width = 1161,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_1 = canvas.create_image(
    581.0,
    473.0,
    image=image_image_1,
    anchor='center'
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda :alg(),  # Pass the alg function directly
    relief="flat"
)
button_1.place(
    x=309.0,
    y=826.0,
    width=543.0,
    height=70.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    581.0,
    377.0,
    image=image_image_2
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: button_2_func(),
    relief="flat"
)
button_2.place(
    x=512.0,
    y=711.0,
    width=139.0,
    height=30.0
)
window.resizable(False, False)
window.mainloop()



