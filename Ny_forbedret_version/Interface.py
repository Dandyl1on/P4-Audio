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

# getroot functions passes the root variable into the fullimage function needed to display a full scale version of the image
def getroot():
    func_for_interface.fullimage(root)

# File is the original sound file chosen by the user
File = None

# Garbage collection will be referenced multiple times below. It is pythons was to improve memory and reduce items that are not used, it mainly happens to tkinters PhotoImage
# None types for Image displaying to not be garbage collected
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
    global PlaceImage
    global File
    global originalImage

    # The filedialog.askopenfilename ask the user to choose a .png or all files to open in the program
    # The file has to be in Ny_forbedret_version due to func_for_inteface´s only works inside that folder if playing the sound
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio/Ny_forbedret_version", title="select a file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*"))
    )
    # Makes the filename into File variable, so it can be used by other functions that isnÂ´t tkinter
    File = os.path.basename(root.filename)
    # Sets the audio_path in Equalloudness_transformation to the file
    Equalloudness_transformation.audio_path = root.filename
    # infoFunc needs to be run in order to pass information about the choosen wav file to other scipts (all filters)
    Equalloudness_transformation.infoFunc()
    # Destroys a placeholder imagelabel to make place for the actual image
    ImageLabel.destroy()
    # Runs the displayimage function from func_for_interface which returns the LoadImage variable
    displayimage()
    # Makes the title of the frame the name of the wav file
    FilteredImage.config(text=File)

    # Preview
    # Makes sure the LoadImage variable can be displayed into a Label and not garbage collected
    # This one with PlaceImage is for the preview
    if PlaceImage is None:
        PlaceImage = Label(UnderFrame, image=func_for_interface.LoadImage)
        PlaceImage.pack()
    # ensures that the user can choose other images and display that image
    else:
        PlaceImage.config(image=func_for_interface.LoadImage)

    # Filter frame
    # Makes sure the LoadImage variable can be displayed into a Label and not garbage collected
    # This one with originalImage is for the image without any filter changes
    # ensures that the user can choose other images and display that image
    if originalImage is not None:
        originalImage.config(image=func_for_interface.Displayed)
    else:
        originalImage = Label(FilterLabel, image=func_for_interface.Displayed)
        originalImage.pack()

    # Returns buttons to normal state so users can press them
    PlayImage.config(state=NORMAL)
    FullImage.config(state=NORMAL)
    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=NORMAL)
    SaveImage.config(state=NORMAL)
    UniPlay.config(state=NORMAL)


# Play music passes the File variable to the play function inside func_for_interface
def playmusic():
    func_for_interface.play(File)

# Creates the bandpass filter frame, sliders and buttons
# Creates the bandpass filter frame, sliders and buttons
def createbandpassframe():
    global BandpassFrame
    global BandBtn
    global NotchBtn

    # ensures the BandLowSlider and BandHighSlider cannot go past each other (Currently not working correctly)
    def lowsliderupdate(val):
        lowlimit = float(BandLowSlider.get())
        highlimit = float(BandHighSlider.get())

        # Ensure the low slider value is less than the high slider
        if lowlimit >= highlimit:
            BandLowSlider.set(high - 1)  # Ensure low is less than high

    # ensures the BandLowSlider and BandHighSlider cannot go past each other (Currently not working correctly)
    def highsliderupdate(val):
        lowlimit = float(BandLowSlider.get())
        highlimit = float(BandHighSlider.get())

        # Ensure the high slider value is greater than the low slider
        if highlimit <= lowlimit:
            BandHighSlider.set(low + 1)  # Ensure high is greater than low

    # Tkinter code for creating the frame and sliders
    BandpassFrame = LabelFrame(FilterFrame, text="Bandpass", font="BOLD")
    BandpassFrame.pack()

    BandLowSlider = Scale(BandpassFrame, from_=0, to=11025, orient=HORIZONTAL, length=200, command=lowsliderupdate)
    BandLowSlider.pack(padx=5, pady=5)
    Label2 = Label(BandpassFrame, text="Adjust low cutoff frequency")
    Label2.pack()

    BandHighSlider = Scale(BandpassFrame, from_=0, to=11025, orient=HORIZONTAL, length=200, command=highsliderupdate)
    BandHighSlider.pack(padx=5, pady=5)
    Label1 = Label(BandpassFrame, text="Adjust high cutoff frequency")
    Label1.pack()

    def getHighandLow():
        # accesses the variable for displaying purposes (was none to avoid garbage collection)
        global BandpassImageDisplay
        global NotchImageDisplay
        global EqualizationImageDisplay
        global FilterLabel

        # Passes the current values of the Sliders to the bandimage function to create the bandpass image
        low = BandLowSlider.get()
        high = BandHighSlider.get()
        bandimage(high, low)

        # Removes the labels displaying the image and text
        originalImage.config(image=func_for_interface.FLoad)

    # Tkinter code for creating the buttons
    Apply = Button(BandpassFrame, text="Apply filter", command=getHighandLow)
    Apply.pack()

    # destroys the empty label that makes the size of the "choose filters" frame
    EmptyLabelFilter.destroy()

    # Changes button states to be normal and disabled so the user cannot create multiple of the same frame
    BandBtn.config(state=DISABLED)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=NORMAL)

    # destroys any of the previous frames
    if EqualFrame or NotchFrame is not None:
        destroynotchframe()
        destroyequalframe()


def createnotchframe():
    global NotchFrame

    NotchFrame = LabelFrame(FilterFrame, text="Notch", font="bold")
    NotchFrame.pack()

    NotchSlider = Scale(NotchFrame, from_=0, to=11025, orient=HORIZONTAL, length=200)
    NotchSlider.pack(padx=5, pady=5)
    Label1 = Label(NotchFrame, text="Adjust Notch")
    Label1.pack()
    BandwidthSlider = Scale(NotchFrame, from_=0, to=11025, orient=HORIZONTAL, length=200)
    BandwidthSlider.pack(padx=5, pady=5)
    Label2 = Label(NotchFrame, text="Adjust Bandwidth")
    Label2.pack()

    def getnotchandbandwidth():
        global NotchImageDisplay
        global EqualizationImageDisplay
        global BandpassImageDisplay
        global FilterLabel

        notch = NotchSlider.get()
        bandwidth = BandwidthSlider.get()
        notchimage(notch, bandwidth)

        originalImage.config(image=func_for_interface.FLoad)

    Apply = Button(NotchFrame, text="Apply filter", command=getnotchandbandwidth)
    Apply.pack()


    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=DISABLED)
    EqualBtn.config(state=NORMAL)

    EmptyLabelFilter.destroy()

    if EqualFrame or BandpassFrame is not None:
        destroybandpassframe()
        destroyequalframe()

def createequalizationframe():
    global EqualFrame

    EqualFrame = LabelFrame(FilterFrame, text="Equalization", font="bold")
    EqualFrame.pack()

    BassSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    BassSlider.grid(row=0, column=0)
    Label1 = Label(EqualFrame, text="Adjust low frequencies")
    Label1.grid(row=1, column=0)
    Range1 = Scale(EqualFrame, from_=0, to=5000, orient=HORIZONTAL)
    Range1.grid(row=0, column=1)
    Range1Label = Label(EqualFrame, text="Adjust range 1")
    Range1Label.grid(row=1, column=1)

    MidSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    MidSlider.grid(row=2, column=0)
    Label2 = Label(EqualFrame, text="Adjust mid frequencies")
    Label2.grid(row=3, column=0)
    Range2 = Scale(EqualFrame, from_=0, to=5000, orient=HORIZONTAL)
    Range2.grid(row=2, column=1)
    Range2Label = Label(EqualFrame, text="Adjust range 2")
    Range2Label.grid(row=3, column=1)

    UppermidSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    UppermidSlider.grid(row=4, column=0)
    Label2 = Label(EqualFrame, text="Adjust uppermid frequencies")
    Label2.grid(row=5, column=0)
    Range3 = Scale(EqualFrame, from_=0, to=5000, orient=HORIZONTAL)
    Range3.grid(row=4, column=1)
    Range3Label = Label(EqualFrame, text="Adjust range 3")
    Range3Label.grid(row=5, column=1)

    HigherSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, length=200, resolution=0.1)
    HigherSlider.grid(row=6, column=0)
    Label2 = Label(EqualFrame, text="Adjust high frequencies")
    Label2.grid(row=7, column=0)
    Range4 = Scale(EqualFrame, from_=0, to=5000, orient=HORIZONTAL)
    Range4.grid(row=6, column=1)
    Range3Label = Label(EqualFrame, text="Adjust range 4")
    Range3Label.grid(row=7, column=1)

    def getequalizationsliders():
        global EqualizationImageDisplay
        global BandpassImageDisplay
        global NotchImageDisplay
        global FilterLabel

        bass = BassSlider.get()
        mid = MidSlider.get()
        uppermid = UppermidSlider.get()
        high = HigherSlider.get()
        range1 = Range1.get()
        range2 = Range2.get()
        range3 = Range3.get()
        range4 = Range4.get()
        equalimage(bass, mid, uppermid, high, range1, range2, range3, range4)

        originalImage.config(image=func_for_interface.FLoad)

    Apply = Button(EqualFrame, text="Apply filter", command=getequalizationsliders)
    Apply.grid(row=4, column=2)

    EmptyLabel3.config(width=10)

    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=DISABLED)

    EmptyLabelFilter.destroy()

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
EmptyLabel1 = Label(Overframe, width=20, height=2, background="grey35")
EmptyLabel1.grid(row=0, column=4)
EmptyLabel2 = Label(Overframe, width=3, height=2, background="grey35")
EmptyLabel2.grid(row=0, column=0)
EmptyLabel3 = Label(Overframe, width=20, height=2, background="grey35")
EmptyLabel3.grid(row=0, column=2)

# Image procsses frame
Processes = LabelFrame(Overframe, text="Choose Image proccessing filter", pady=5, padx=5)
Processes.grid(row=2, column=1)

EmptyLabelProcess = Label(Processes, text="", width=31, height=10)
EmptyLabelProcess.pack(side=TOP)

BtnFrameProces = Frame(Processes, pady=5)
BtnFrameProces.pack(side=BOTTOM)

NoiseReduc = Button(BtnFrameProces, text="Noise reduction filter")
NoiseReduc.grid(row=0, column=0, padx=5)

Sharp = Button(BtnFrameProces, text="Sharpening filter")
Sharp.grid(row=0, column=1, padx=5)

Smooth = Button(BtnFrameProces, text="Smoothing filter")
Smooth.grid(row=0, column=2, padx=5)

# Filter frame
FilterFrame = LabelFrame(Overframe, text="Choose filters", padx=5, pady=5)
FilterFrame.grid(row=1, column=1)

EmptyLabelFilter = Label(FilterFrame, text="", width=31, height=10)
EmptyLabelFilter.pack(side=TOP)

BtnFrame = Frame(FilterFrame, pady=5)
BtnFrame.pack(side=BOTTOM)

BandBtn = Button(BtnFrame, text="Bandpass Filter", command=createbandpassframe)
BandBtn.grid(row=0, column=0, padx=5)
BandBtn.config(state=DISABLED)

NotchBtn = Button(BtnFrame, text="Notch Filter", command=createnotchframe)
NotchBtn.grid(row=0, column=1, padx=5)
NotchBtn.config(state=DISABLED)

EqualBtn = Button(BtnFrame, text="Equalization Filter", command=createequalizationframe)
EqualBtn.grid(row=0, column=2, padx=5)
EqualBtn.config(state=DISABLED)

# Preview frame
DisplayFrame = LabelFrame(Overframe, text="Preview", pady=5, padx=5)
DisplayFrame.grid(row=1, column=5)

UnderFrame = Frame(DisplayFrame)
UnderFrame.pack()

BtnFrame = Frame(UnderFrame, pady=5, padx=5)
BtnFrame.pack(side=BOTTOM)
InstancesFrame = Frame(DisplayFrame, pady=5, padx=5)
InstancesFrame.pack(side=BOTTOM)

ImageLabel = Label(UnderFrame, text="Your image will be displayed", width=42, height=24)
ImageLabel.pack(side=TOP)

PlayImage = Button(BtnFrame, text="Play sound", pady=5, padx=15, command=playmusic)
PlayImage.grid(row=0, column=0)
PlayImage.config(state=DISABLED)


SaveImage = Button(BtnFrame, text="Save Image", padx=5, pady=5, command=saveimage)
SaveImage.grid(row=0, column=1)
SaveImage.config(state=DISABLED)

FullImage = Button(BtnFrame, text="Show full image", pady=5, padx=5, command=getroot)
FullImage.grid(row=0, column=2)
FullImage.config(state=DISABLED)

# Small Images
SmallImages = LabelFrame(InstancesFrame, text="Instances")
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

# Universal frame
UniversalFrame = Frame(Overframe, width=600, padx=5, pady=5)
UniversalFrame.grid(row=2, column=3)

UniPlay = Button(UniversalFrame, text="Play Image", command=func_for_interface.playfilter)
UniPlay.pack()
UniPlay.config(state=DISABLED)

mainloop()