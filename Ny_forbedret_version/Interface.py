import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import PIL.Image
import cv2
import pygame
import time
import soundfile as sf

import func_for_interface
from func_for_interface import *

import Equalloudness_transformation
from Equalloudness_transformation import *

import all_filters
from PIL.ImageOps import scale
from all_filters import ApplyFilters

import Equalization
from Equalization import *

from PIL.ImageFilter import Kernel
from scipy.io import wavfile
from PIL import Image, ImageTk
from fileinput import filename
from tkinter import *
from tkinter import filedialog, Label, Tk, messagebox as mb, ttk

# Placement and list of saved images intances
Placement = 0
photo_image_references = []

# Creates the window
root = Tk()
root.title("Modifun")
root.geometry("700x600")

def getroot():
    func_for_interface.fullimage(root)

# File is the original sound file chosen by the user
File = None
# CV2Image is the image made from the Equalloudness transformation script, so it can be opened in tkinter
CV2Image = None

# None types for Image displaying
PlaceImage = None
SImage = None
SmallImageLoad = None

BandpassImageDisplay = None
NotchImageDisplay = None
EqualizationImageDisplay = None

# StartImage ensures that the unfilted opened image can be created and not garbage collected
originalImage = None
Full = None

# None types for filter sliders
BandpassFrame = None
NotchFrame = None
EqualFrame = None



# Creates a sound player from pygame
pygame.mixer.init()

def selectimage():
    global File
    global LoadImage
    global PlaceImage
    global File
    global CV2Image
    global ogImage
    global originalImage
    # Calls stop function to unload any previous sounds played
    stop()

    # The filedialog.askopenfilename ask the user to choose a .png or all files to open in the program
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio/Ny_forbedret_version", title="select a file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*"))
    )
    # Makes the filename into File variable, so it can be used by other functions that isn´t tkinter
    File = os.path.basename(root.filename)
    # Sets the audio_path in Equalloudness_transformation to the file
    Equalloudness_transformation.audio_path = root.filename

    print(Equalloudness_transformation.audio_path)
    Equalloudness_transformation.infoFunc()

    ImageLabel.destroy()
    displayimage()
    FilteredImage.config(text=File)

    if PlaceImage is None:
        PlaceImage = Label(UnderFrame, image=func_for_interface.LoadImage)
        PlaceImage.pack()
    else:
        PlaceImage.config(image=func_for_interface.LoadImage)

    ogImage = func_for_interface.LoadImage

    if originalImage is not None:
        originalImage.config(image=ogImage)
    else:
        originalImage = Label(FilterLabel, image=ogImage)
        originalImage.pack()

    PlayImage.config(state=NORMAL)
    FullImage.config(state=NORMAL)
    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=NORMAL)
    SaveImage.config(state=NORMAL)
    # PlacingNumbers()

# Play music passes the File variable to the play function inside func_for_interface
def playmusic():
    func_for_interface.play(File)

def createbandpassframe():
    global BandpassFrame
    global BandBtn
    global NotchBtn
    global BandHighSlider
    global BandLowSlider

    def lowsliderupdate(val):
        lowlimit = float(BandLowSlider.get())
        highlimit = float(BandHighSlider.get())

        # Ensure the low slider value is less than the high slider
        if lowlimit >= highlimit:
            BandLowSlider.set(high - 1)  # Ensure low is less than high

    def highsliderupdate(val):
        lowlimit = float(BandLowSlider.get())
        highlimit = float(BandHighSlider.get())

        # Ensure the high slider value is greater than the low slider
        if highlimit <= lowlimit:
            BandHighSlider.set(low + 1)  # Ensure high is greater than low

    BandpassFrame = LabelFrame(FilterFrame, text="Bandpass", font="BOLD")
    BandpassFrame.pack()

    BandLowSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200, command=lowsliderupdate)
    BandLowSlider.pack(padx=5, pady=5)
    Label2 = Label(BandpassFrame, text="Adjust lowcut frequency")
    Label2.pack()

    BandHighSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200, command=highsliderupdate)
    BandHighSlider.pack(padx=5, pady=5)
    Label1 = Label(BandpassFrame, text="Adjust highcut frequency")
    Label1.pack()

    def getHighandLow():
        global BandpassImageDisplay
        global FLoad
        global CV2Image
        global FilterLabel

        low = BandLowSlider.get()
        high = BandHighSlider.get()
        bandimage(high, low)

        FilterLabel.destroy()
        originalImage.destroy()

        if BandpassImageDisplay is None:
            BandpassImageDisplay = Label(FilteredImage, image=func_for_interface.FLoad)
            BandpassImageDisplay.pack(side=RIGHT)
        else:
            BandpassImageDisplay.config(image=func_for_interface.FLoad)

    Apply = Button(BandpassFrame, text="Apply filter", command=getHighandLow)
    Apply.pack(side=LEFT)
    Apply = Button(BandpassFrame, text="Play sound", command=playfilter)
    Apply.pack(side=RIGHT)

    BandBtn.config(state=DISABLED)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=NORMAL)

    L.destroy()

    if EqualFrame or NotchFrame is not None:
        destroynotchframe()
        destroyequalframe()

def createnotchframe():
    global NotchFrame
    global NotchSlider
    global BandwidthSlider

    NotchFrame = LabelFrame(FilterFrame, text="Notch", font="bold")
    NotchFrame.pack()

    NotchSlider = Scale(NotchFrame, from_=0, to=22050, orient=HORIZONTAL, length=200)
    NotchSlider.pack(padx=5, pady=5)
    Label1 = Label(NotchFrame, text="Adjust Notch")
    Label1.pack()
    BandwidthSlider = Scale(NotchFrame, from_=0, to=22050, orient=HORIZONTAL, length=200)
    BandwidthSlider.pack(padx=5, pady=5)
    Label2 = Label(NotchFrame, text="Adjust Bandwidth")
    Label2.pack()

    def getnotchandbandwidth():
        global NotchImageDisplay
        global FLoad
        global CV2Image
        global FilterLabel

        notch = NotchSlider.get()
        bandwidth = BandwidthSlider.get()
        notchimage(notch, bandwidth)

        FilterLabel.destroy()
        originalImage.destroy()

        if NotchImageDisplay is None:
            NotchImageDisplay = Label(FilteredImage, image=func_for_interface.FLoad)
            NotchImageDisplay.pack(side=RIGHT)
        else:
            NotchImageDisplay.config(image=func_for_interface.FLoad)

    Apply = Button(NotchFrame, text="Apply filter", command=getnotchandbandwidth)
    Apply.pack(side=LEFT)
    Apply = Button(NotchFrame, text="Play sound", command=playfilter)
    Apply.pack(side=RIGHT)

    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=DISABLED)
    EqualBtn.config(state=NORMAL)

    L.destroy()

    if EqualFrame or BandpassFrame is not None:
        destroybandpassframe()
        destroyequalframe()

def createequalizationframe():
    global EqualFrame
    global BassSlider
    global MidSlider
    global UppermidSlider
    global HigherSlider

    EqualFrame = LabelFrame(FilterFrame, text="Equalization", font="bold")
    EqualFrame.pack()

    BassSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    BassSlider.pack(padx=5, pady=5)
    Label1 = Label(EqualFrame, text="Adjust low frequencies")
    Label1.pack()
    MidSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    MidSlider.pack(padx=5, pady=5)
    Label2 = Label(EqualFrame, text="Adjust mid frequencies")
    Label2.pack()
    UppermidSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    UppermidSlider.pack(padx=5, pady=5)
    Label2 = Label(EqualFrame, text="Adjust uppermid frequencies")
    Label2.pack()
    HigherSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    HigherSlider.pack(padx=5, pady=5)
    Label2 = Label(EqualFrame, text="Adjust high frequencies")
    Label2.pack()

    def getequalizationsliders():
        global EqualizationImageDisplay
        global FLoad
        global CV2Image
        global FilterLabel

        bass = BassSlider.get()
        mid = MidSlider.get()
        uppermid = UppermidSlider.get()
        high = HigherSlider.get()
        equalimage(bass, mid, uppermid, high)

        FilterLabel.destroy()
        originalImage.destroy()

        if EqualizationImageDisplay is None:
            EqualizationImageDisplay = Label(FilteredImage, image=func_for_interface.FLoad)
            EqualizationImageDisplay.pack(side=RIGHT)
        else:
            EqualizationImageDisplay.config(image=func_for_interface.FLoad)

    Apply = Button(EqualFrame, text="Apply filter", command=getequalizationsliders)
    Apply.pack(side=LEFT)
    Apply = Button(EqualFrame, text="Play sound", command=playfilter)
    Apply.pack(side=RIGHT)

    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=DISABLED)

    L.destroy()

    if BandpassFrame or NotchFrame is not None:
        destroybandpassframe()
        destroynotchframe()

def destroybandpassframe():
    global BandpassFrame
    if BandpassFrame is not None:
        BandpassFrame.destroy()
        BandpassFrame = None

def destroynotchframe():
    global NotchFrame
    if NotchFrame is not None:
        NotchFrame.destroy()
        NotchFrame = None

def destroyequalframe():
    global EqualFrame
    if EqualFrame is not None:
        EqualFrame.destroy()
        EqualFrame = None

def saveimage():
    global CV2Image
    global SImage
    global SmallImageLoad
    global File
    global Placement

    SImage = Label(scrollframe)
    SImage.grid(row=Placement, column=0, padx=0)

    text = Label(scrollframe, text=File, padx=0, pady=5, width=25)
    text.grid(row=Placement, column=1)

    Size = func_for_interface.CV2Image.resize((50, 50), PIL.Image.LANCZOS)
    SmallImageLoad = ImageTk.PhotoImage(Size)

    photo_image_references.append(SmallImageLoad)

    SImage.config(image=SmallImageLoad)
    Placement += 1


# Creates a menu in the interface, where the select function is called
menu = Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open sound path", command=selectimage)

Overframe = Frame(root, background="grey35")
Overframe.pack(fill="both", expand=True)

# Empty used for layout mangement
EmptyLabel = Label(Overframe, width=20, height=2, background="grey35")
EmptyLabel.grid(row=0, column=4)
EmptyLabel = Label(Overframe, width=3, height=2, background="grey35")
EmptyLabel.grid(row=0, column=0)
EmptyLabel = Label(Overframe, width=20, height=2, background="grey35")
EmptyLabel.grid(row=0, column=2)

# Filter frame
FilterFrame = LabelFrame(Overframe, text="Choose filters", padx=5, pady=5)
FilterFrame.grid(row=1, column=1)

L = Label(FilterFrame, text="", width=31, height=10)
L.pack(side=TOP)

BtnFrame = Frame(FilterFrame, pady=5)
BtnFrame.pack(side=BOTTOM)

BandBtn = Button(BtnFrame, text="Bandpass Filter", command=createbandpassframe)
BandBtn.grid(row=0, column=0, padx=5)
#BandBtn.config(state=DISABLED)
NotchBtn = Button(BtnFrame, text="Notch Filter", command=createnotchframe)
NotchBtn.grid(row=0, column=1, padx=5)
#NotchBtn.config(state=DISABLED)
EqualBtn = Button(BtnFrame, text="Equalization Filter", command=createequalizationframe)
EqualBtn.grid(row=0, column=2, padx=5)
#EqualBtn.config(state=DISABLED)

# Display frame
DisplayFrame = LabelFrame(Overframe, text="Preview", pady=5, padx=5)
DisplayFrame.grid(row=1, column=5)

UnderFrame = Frame(DisplayFrame)
UnderFrame.pack()

BtnFrame = Frame(UnderFrame, pady=5, padx=5)
BtnFrame.pack(side=BOTTOM)
F = Frame(DisplayFrame, pady=5, padx=5)
F.pack(side=BOTTOM)

ImageLabel = Label(UnderFrame, text="Your image will be displayed here", width=42, height=24)
ImageLabel.pack(side=TOP)

PlayImage = Button(BtnFrame, text="Play sound", pady=5, padx=15, command=playmusic)
PlayImage.grid(row=0, column=0)
#PlayImage.config(state=DISABLED)


SaveImage = Button(BtnFrame, text="Save Image", padx=5, pady=5, command=saveimage)
SaveImage.grid(row=0, column=1)
#SaveImage.config(state=DISABLED)

FullImage = Button(BtnFrame, text="Show full image", pady=5, padx=5, command=getroot)
FullImage.grid(row=0, column=2)
#FullImage.config(state=DISABLED)

# Small Images
SmallImages = LabelFrame(F, text="Instances")
SmallImages.grid(row=2, column=4)

canvas = Canvas(SmallImages, height=70, width=260)
canvas.grid(row=0, column=0, sticky="nsew")

scroll = Scrollbar(SmallImages, command=canvas.yview)
scroll.grid(row=0, column=1, sticky="ns")

scrollframe = Frame(canvas, pady=5, padx=0)

canvas.create_window((0, 0), window=scrollframe, anchor="nw")
canvas.configure(yscrollcommand=scroll.set)

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollframe.bind("<Configure>", on_frame_configure)

SmallImages.grid_rowconfigure(0, weight=1)
SmallImages.grid_columnconfigure(0, weight=1)

# Middle display frame
FilteredImage = LabelFrame(Overframe, text="Image with choosen filter applied", width=600, height=500)
FilteredImage.grid(row=1, column=3)

FilterLabel = Label(FilteredImage, text="Your image will be displayed here", width=78, height=32)
FilterLabel.pack(side=RIGHT)

NumberFrame = Frame(FilteredImage, height=32)
NumberFrame.pack()

mainloop()