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

root = Tk()
root.title("App prototype")
root.geometry("500x500")

LoadImage = None
filterImage = None
transformed_image = None

pygame.mixer.init()

def select():
    mb.showinfo("Choose", "Choose a picture you want to apply filters to")

    global LoadImage
    global DisplayImage
    global filterImage
    global CV2Image
    global transformed_image
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file",
        filetypes=(("PNG files", "*.png"), ("All files", "*"))
    )

    filterImage = root.filename

    CV2Image = cv2.imread(root.filename, cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    LoadImage = ImageTk.PhotoImage(CV2Image)
    DisplayImage = Label(ImageFrame, image=LoadImage)
    DisplayImage.pack()

    mb.showinfo("Sliders", "Use the sliders on the left to adjust the filter parameters")
    print(filterImage)


def Apply():
    global filterImage
    global CV2Image
    global transformed_image
    Ksize = (Slider1.get() | 1, Slider2.get() | 1)
    sigma = Slider3.get()

    GausianImage = cv2.imread(root.filename, cv2.IMREAD_UNCHANGED)
    Gaussian = cv2.GaussianBlur(GausianImage, Ksize, sigma)

    if Gaussian.dtype == np.uint16:
        Gaussian = (Gaussian / 256).astype('uint8')
    CV2Image = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    transformed_image = ImageTk.PhotoImage(CV2Image)

    Test = Label(ImageFrame, image=transformed_image)
    Test.image = transformed_image
    Test.pack()


def callback():
    if mb.askyesno("Verify", "Do you really want to Exit?"):
        mb.showwarning("Yes","Exit not yet implemented")
    else:
        mb.showinfo("No","Exit has been cancelled")

def Play():
    pygame.mixer.music.load("Image_to_Audio.wav")
    pygame.mixer.music.play(loops=0)




menu= Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open...", command=select)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=callback)

OverFrame = Frame(root, pady=5, padx=5, height=500)
OverFrame.pack()

SliderFrame = LabelFrame(OverFrame,text="Adjust these sliders", pady=50, padx=5, height=300)
SliderFrame.grid(row=0, column=0)

Slider1=Scale(SliderFrame, from_=0, to=200, orient=HORIZONTAL, troughcolor="darkgreen", background="green")
Slider1.grid(row=0, column=0)
Slider2=Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL, troughcolor="darkblue", background="blue")
Slider2.grid(row=1, column=0)
Slider3=Scale(SliderFrame, from_=0, to=50, tickinterval=15, orient=HORIZONTAL, troughcolor="darkred", background="red")
Slider3.grid(row=2, column=0)

Btn = Button(SliderFrame, text="Apply filter", command=Apply)
Btn.grid(row=0, column=1)

ImageFrame = LabelFrame(OverFrame, text="This is the image", padx=5, pady=5)
ImageFrame.grid(row=0, column=1)

SoundFrame = LabelFrame(OverFrame, text="Hear the picture here", padx=5, pady=5)
SoundFrame.grid(row=1, column=0)

SoundBtn = Button(SoundFrame, text="Play Picture", command=Play)
SoundBtn.pack()

mainloop()