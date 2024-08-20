import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import PIL.Image
import cv2
import pygame
import main
from main import *

from PIL.ImageFilter import Kernel
from scipy.io import wavfile
from PIL import Image, ImageTk
from fileinput import filename
from tkinter import *
from tkinter import filedialog, Label, Tk, messagebox as mb, ttk

root = Tk()
root.title("Modifun")
root.geometry("700x600")

LoadImage = None
BandIMG = None

# Creates a sound player from pygame
pygame.mixer.init()

def select():
    global Filepath
    stop()
    # The filedialog.askopenfilename ask the user to choose a .png or all files to open in the program
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*"))
    )
    Filepath.delete(0, END)
    Filepath.insert(0, root.filename)

    main.audio_path = root.filename
    print(main.audio_path)

    main.mainfunc()
    progressbar.step(99)


def play():
    # Plays the sound in the load method
    pygame.mixer.music.load('reconstructed_audio_from_combined_image.wav')
    pygame.mixer.music.play(loops=0)
    progressbar.step(-99)

def stop():
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()


def getfilepath():
    return Filepath.get()

def displayimage():

    global LoadImage

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    LoadImage = ImageTk.PhotoImage(CV2Image)
    PlaceImage = Label(DisplayFrame, image=LoadImage)
    PlaceImage.grid(row=0, column=0)
    ImageLabel.destroy()

def createbandpass():
    global Bandpass
    global ImageFrameBand
    global BandImage
    Bandpass = Toplevel(root)
    Bandpass.title("Bandpass filter")
    Bandpass.geometry("500x500")
    BandpassFrame = Frame(Bandpass)
    BandpassFrame.pack()

    menu = Menu(Bandpass)
    Bandpass.config(menu=menu)
    BandMenu = Menu(menu)
    menu.add_cascade(label="File", menu=BandMenu)
    BandMenu.add_command(label="Exit", command=Exit)

    ImageFrameBand = LabelFrame(BandpassFrame, text="BandPassImage", pady=5, padx=5)
    ImageFrameBand.grid(row=0, column=0)

    BandImage = Button(ImageFrameBand, text="Show image", pady=5, padx=5, command=BandImage)
    BandImage.grid(row=1, column=0)

    SliderFrame = LabelFrame(BandpassFrame, text="Adjust sliders", padx=5, pady=5)
    SliderFrame.grid(row=1, column=0)

    Slider1 = Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL)
    Slider1.grid(row=0, column=0)
    Slider2 = Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL)
    Slider2.grid(row=0, column=1)
    Slider3 = Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL)
    Slider3.grid(row=1, column=0)
    Slider4 = Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL)
    Slider4.grid(row=1, column=1)


def Exit():
    Bandpass.destroy()

def BandImage():

    global BandIMG

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    BandIMG = ImageTk.PhotoImage(CV2Image)
    PlaceImage = Label(ImageFrameBand, image=BandIMG)
    PlaceImage.grid(row=0, column=0)

# Creates a menu in the interface, where the select function is called
menu = Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open sound path", command=select)
fileMenu.add_separator()
fileMenu.add_command(label="Bandpass filter", command=createbandpass)

Overframe = Frame(root)
Overframe.pack()

# Entry frame
EntryFrame = LabelFrame(Overframe, text="Your sounds path directory")
EntryFrame.grid(row=0, column=0)

Playsound = Button(EntryFrame, text="Play sound", pady=5, padx=5, command=play)
Playsound.grid(row=1, column=0)

Filepath = Entry(EntryFrame, width=80)
Filepath.grid(row=0, column=0)

progressbar = ttk.Progressbar(Overframe, orient=HORIZONTAL)
progressbar.grid(row=1, column=0)


# Display frame
DisplayFrame = LabelFrame(Overframe, text="Here is the Image", pady=5, padx=5)
DisplayFrame.grid(row=2, column=0)

ImageLabel = Label(DisplayFrame, text="Your image will be displayed here")
ImageLabel.grid(row=0, column=0)

ShowImage = Button(DisplayFrame, text="Show image", pady=5, padx=5, command=displayimage)
ShowImage.grid(row=1, column=0)


mainloop()