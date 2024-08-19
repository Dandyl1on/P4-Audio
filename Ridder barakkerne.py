from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

from PIL.ImageFilter import Kernel
from scipy.io import wavfile
from PIL import Image, ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog, Label, Tk, messagebox as mb
import cv2
import pygame

# Creates the interface window
root = Tk()
root.title("App prototype")
root.geometry("700x600")

# Public variables used for images currently none, so it can be filled with choosen images
LoadImage = None
transformed_image = None

# Creates a sound player from pygame
pygame.mixer.init()

def select():
    # Messagebox with the defined text
    mb.showinfo("Choose", "Choose a picture you want to apply filters to")
    # All global variables can be used in different functions
    global LoadImage
    global CV2Image

    # The filedialog.askopenfilename ask the user to choose a .png or all files to open in the program
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file",
        filetypes=(("PNG files", "*.png"), ("All files", "*"))
    )

    # A series of code that converts the image to a cv2 image and then into a PIL image, so it can be displayed in TKinter
    CV2Image = cv2.imread(root.filename, cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    LoadImage = ImageTk.PhotoImage(CV2Image)

    # Takes the now Tkinter photoImage and loads it into a label inside the ImageFrame
    DisplayImage = Label(ImageFrame, image=LoadImage)
    DisplayImage.pack()

    # Messagebox with the defined text
    mb.showinfo("Sliders", "Use the sliders on the left to adjust the filter parameters")


def apply():
    # All global variables can be used in different functions
    global CV2Image
    global transformed_image

    # The gaussianblur kernel size is defined by the silder1 and slider2 values. The | 1 ensures it's always an odd number.
    Ksize = (Slider1.get() | 1, Slider2.get() | 1)
    # Sigma is defined by slider3 values
    sigma = Slider3.get()

    # Applies gaussianblur to the choosen image from the select function with the values from Slider 1,2 and 3
    GausianImage = cv2.imread(root.filename, cv2.IMREAD_UNCHANGED)
    Gaussian = cv2.GaussianBlur(GausianImage, Ksize, sigma)
    # A series of code that converts the image to a cv2 image and then into a PIL image, so it can be displayed in TKinter
    if Gaussian.dtype == np.uint16:
        Gaussian = (Gaussian / 256).astype('uint8')
    CV2Image = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    transformed_image = ImageTk.PhotoImage(CV2Image)
    # Takes the now Tkinter Transformed_image and loads it into a label inside the ImageFrame
    NewImage = Label(ImageFrame, image=transformed_image)
    NewImage.image = transformed_image
    NewImage.pack()

def callback():
    # Exit function
    if mb.askyesno("Verify", "Do you really want to Exit?"):
        mb.showwarning("Yes", "Exit not yet implemented")
    else:
        mb.showinfo("No", "Exit has been cancelled")

def play():
    # Plays the sound in the load method
    pygame.mixer.music.load("Image_to_Audio.wav")
    pygame.mixer.music.play(loops=0)

# Creates a menu in the interface, where the select function is called
menu = Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open...", command=select)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=callback)

# Creates an "Overframe" that stores all the other frames
OverFrame = Frame(root, pady=5, padx=5, height=500)
OverFrame.pack()

# A frame for the sliders
SliderFrame = LabelFrame(OverFrame, text="Adjust these sliders", pady=50, padx=5, height=300)
SliderFrame.grid(row=0, column=0)

# Sliders
Slider1 = Scale(SliderFrame, from_=0, to=200, orient=HORIZONTAL, troughcolor="darkgreen", background="green")
Slider1.grid(row=0, column=0)
Slider2 = Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL, troughcolor="darkblue", background="blue")
Slider2.grid(row=1, column=0)
Slider3 = Scale(SliderFrame, from_=0, to=50, tickinterval=15, orient=HORIZONTAL, troughcolor="darkred", background="red")
Slider3.grid(row=2, column=0)

# Button to call the Apply function with
Btn = Button(SliderFrame, text="Apply filter", command=apply)
Btn.grid(row=0, column=1)

# Creates the ImageFrame to be inside the OverFrame
ImageFrame = LabelFrame(OverFrame, text="This is the image", padx=5, pady=5)
ImageFrame.grid(row=0, column=1)

# Creates a SoundFrame
SoundFrame = LabelFrame(OverFrame, text="Hear the picture here", padx=5, pady=5)
SoundFrame.grid(row=1, column=0)

# Creates a button the call the play function
SoundBtn = Button(SoundFrame, text="Play Picture", command=play)
SoundBtn.pack()

mainloop()