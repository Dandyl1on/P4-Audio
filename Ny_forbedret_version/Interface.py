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
from PIL.ImageOps import scale, expand
from all_filters import ApplyFilters

import Equalization
from Equalization import *

import sharpening_filter
from sharpening_filter import *

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

# Garbage collection will be referenced multiple times below. It is pythons way to improve memory and reduce items that are not used, it mainly happens to tkinters PhotoImage
# None types for Image displaying to not be garbage collected
PlaceImage = None
SImage = None
SmallImageLoad = None
SharpImageFinal = None

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

SharpeningFrame = None
Smoothingframe = None
Noiseframe = None

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
    UniPlay.config(state=NORMAL)
    Noise.config(state=NORMAL)
    Sharp.config(state=NORMAL)
    Smooth.config(state=NORMAL)

    if BandpassFrame is not None:
        BandpassFrame.destroy()
    if NotchFrame is not None:
        NotchFrame.destroy()
    if EqualFrame is not None:
        EqualFrame.destroy()


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

    BandLowSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200, command=lowsliderupdate)
    BandLowSlider.pack(padx=5, pady=5)
    Label2 = Label(BandpassFrame, text="Adjust low cutoff frequency")
    Label2.pack()

    BandHighSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200, command=highsliderupdate)
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
    Infobtn = Button(BandpassFrame, text="Information", command=func_for_interface.BandInformation)
    Infobtn.pack()

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
    Infobtn = Button(NotchFrame, text="Information", command=func_for_interface.NotchInformation)
    Infobtn.pack()


    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=DISABLED)
    EqualBtn.config(state=NORMAL)

    if EqualFrame or BandpassFrame is not None:
        destroybandpassframe()
        destroyequalframe()

def createequalizationframe():
    global EqualFrame

    EqualFrame = LabelFrame(FilterFrame, text="Equalization", font="bold")
    EqualFrame.pack()

    BassSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, resolution=0.1)
    BassSlider.grid(row=0, column=0)
    Label1 = Label(EqualFrame, text="Adjust low frequencies")
    Label1.grid(row=1, column=0)
    Band1 = Scale(EqualFrame, from_=0, to=22050, length=200, orient=HORIZONTAL)
    Band1.grid(row=0, column=1)
    Range1Label = Label(EqualFrame, text="Adjust band 1")
    Range1Label.grid(row=1, column=1)

    MidSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, resolution=0.1)
    MidSlider.grid(row=2, column=0)
    Label2 = Label(EqualFrame, text="Adjust mid frequencies")
    Label2.grid(row=3, column=0)
    Band2 = Scale(EqualFrame, from_=0, to=22050, length=200, orient=HORIZONTAL)
    Band2.grid(row=2, column=1)
    Range2Label = Label(EqualFrame, text="Adjust band 2")
    Range2Label.grid(row=3, column=1)

    UppermidSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, resolution=0.1)
    UppermidSlider.grid(row=4, column=0)
    Label2 = Label(EqualFrame, text="Adjust uppermid frequencies")
    Label2.grid(row=5, column=0)
    Band3 = Scale(EqualFrame, from_=0, to=22050, length=200, orient=HORIZONTAL)
    Band3.grid(row=4, column=1)
    Range3Label = Label(EqualFrame, text="Adjust band 3")
    Range3Label.grid(row=5, column=1)

    HigherSlider = Scale(EqualFrame, from_=0, to=2, orient=HORIZONTAL, resolution=0.1)
    HigherSlider.grid(row=6, column=0)
    Label2 = Label(EqualFrame, text="Adjust high frequencies")
    Label2.grid(row=7, column=0)
    Band4 = Scale(EqualFrame, from_=0, to=22050, length=200, orient=HORIZONTAL)
    Band4.grid(row=6, column=1)
    Range3Label = Label(EqualFrame, text="Adjust band 4")
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
        range1 = Band1.get()
        range2 = Band2.get()
        range3 = Band3.get()
        range4 = Band4.get()
        equalimage(bass, mid, uppermid, high, range1, range2, range3, range4)

        originalImage.config(image=func_for_interface.FLoad)

    Apply = Button(EqualFrame, text="Apply filter", command=getequalizationsliders)
    Apply.grid(row=3, column=2)
    Infobtn = Button(EqualFrame, text="Information", command=func_for_interface.EqualizationInformation)
    Infobtn.grid(row=4, column=2)

    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=DISABLED)


    if BandpassFrame or NotchFrame is not None:
        destroybandpassframe()
        destroynotchframe()

def createsharpframe():
    global SharpeningFrame

    SharpeningFrame = LabelFrame(ProcessesFrame, text="Sharpening", font="Bold")
    SharpeningFrame.pack()

    KernelScale = Scale(SharpeningFrame, to=func_for_interface.width, from_=1, orient=HORIZONTAL)
    KernelScale.grid(row=0, column=0)
    KLabel = Label(SharpeningFrame, text="Adjust Kernel")
    KLabel.grid(row=1, column=0)

    SigmaScale = Scale(SharpeningFrame, to=20, from_=0, orient=HORIZONTAL)
    SigmaScale.grid(row=2, column=0)
    SLabel = Label(SharpeningFrame, text="Adjust Sigma")
    SLabel.grid(row=3, column=0)

    AlphaScale = Scale(SharpeningFrame, to=10, from_=0, orient=HORIZONTAL)
    AlphaScale.grid(row=0, column=1)
    ALabel = Label(SharpeningFrame, text="Adjust Alpha")
    ALabel.grid(row=1, column=1)

    BetaScale = Scale(SharpeningFrame, to=5, from_=-5, orient=HORIZONTAL, resolution=0.1)
    BetaScale.grid(row=2, column=1)
    BLabel = Label(SharpeningFrame, text="Adjust Beta")
    BLabel.grid(row=3, column=1)

    GammaScale = Scale(SharpeningFrame, to=10, from_=0, orient=HORIZONTAL)
    GammaScale.grid(row=4, column=1)
    GLabel = Label(SharpeningFrame, text="Adjust Gamma")
    GLabel.grid(row=5, column=1)

    def getvalues():

        Kernelval = KernelScale.get()
        Sigmaval = SigmaScale.get()
        Aplhaval = AlphaScale.get()
        Betaval = BetaScale.get()
        Gammaval = GammaScale.get()

        Sharppathchange(Kernelval, Sigmaval, Aplhaval, Betaval, Gammaval)

        originalImage.config(image=func_for_interface.SharpImageFinal)

    Applybtn = Button(SharpeningFrame, text="Apply", command=getvalues)
    Applybtn.grid(row=4, column=0)
    Infobtn = Button(SharpeningFrame, text="Information", command=func_for_interface.SharpInformation)
    Infobtn.grid(row=5, column=0)

    Noise.config(state=NORMAL)
    Sharp.config(state=DISABLED)
    Smooth.config(state=NORMAL)

    if Smoothingframe or Noiseframe is not None:
        destroysmoothframe()
        destroynoiseframe()



def createsmoothframe():
    global Smoothingframe

    Smoothingframe = LabelFrame(ProcessesFrame, text="Smoothing", font="Bold")
    Smoothingframe.pack()

    kernelScale = Scale(Smoothingframe, to=func_for_interface.width, from_=0, orient=HORIZONTAL)
    kernelScale.grid(row=0, column=0)
    Klabel = Label(Smoothingframe, text="Adjust kernel")
    Klabel.grid(row=1, column=0)

    weightScale = Scale(Smoothingframe, to=5, from_=0, orient=HORIZONTAL, resolution=0.1)
    weightScale.grid(row=0, column=1)
    Wlabel = Label(Smoothingframe, text="Adjust weight")
    Wlabel.grid(row=1, column=1)

    def getvalues():
        Kernelval = kernelScale.get()
        Weightval = weightScale.get()

        Smoothpathchange(Kernelval, Weightval)

        originalImage.config(image=func_for_interface.SmoothImageFinal)

    Applybtn = Button(Smoothingframe, text="Apply", command=getvalues)
    Applybtn.grid(row=2, column=0)
    Infobtn = Button(Smoothingframe, text="Information", command=func_for_interface.SmoothInformation)
    Infobtn.grid(row=2, column=1)

    Noise.config(state=NORMAL)
    Sharp.config(state=NORMAL)
    Smooth.config(state=DISABLED)

    if SharpeningFrame or Noiseframe is not None:
        destroysharpframe()
        destroynoiseframe()

def createnoiseframe():
    global Noiseframe

    Noiseframe = LabelFrame(ProcessesFrame, text="Noise", font="Bold")
    Noiseframe.pack()

    meanScale = Scale(Noiseframe, to=10, from_=0, orient=HORIZONTAL)
    meanScale.grid(row=0, column=0)
    MLabel = Label(Noiseframe, text="Adjust Mean")
    MLabel.grid(row=1, column=0)

    sigmaScale = Scale(Noiseframe, to=20, from_=0, orient=HORIZONTAL)
    sigmaScale.grid(row=0, column=1)
    SLabel = Label(Noiseframe, text="Adjust Sigma")
    SLabel.grid(row=1, column=1)

    def getvalues():
        Meanval = meanScale.get()
        Sigmaval = sigmaScale.get()

        Noisepathchange(Meanval, Sigmaval)

        originalImage.config(image=func_for_interface.NoiseImageFinal)

    Applybtn = Button(Noiseframe, text="Apply", command=getvalues)
    Applybtn.grid(row=2, column=0)
    Infobtn = Button(Noiseframe, text="Information", command=func_for_interface.NoiseInformation)
    Infobtn.grid(row=2, column=1)

    Noise.config(state=DISABLED)
    Sharp.config(state=NORMAL)
    Smooth.config(state=NORMAL)

    if SharpeningFrame or Smoothingframe is not None:
        destroysharpframe()
        destroysmoothframe()

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
def destroysharpframe():
    global SharpeningFrame
    if SharpeningFrame is not None:
        SharpeningFrame.destroy()
        SharpeningFrame = None
def destroysmoothframe():
    global Smoothingframe
    if Smoothingframe is not None:
        Smoothingframe.destroy()
        Smoothingframe = None
def destroynoiseframe():
    global Noiseframe
    if Noiseframe is not None:
        Noiseframe.destroy()
        Noiseframe = None


# Creates a menu in the interface, where the select function is called
menu = Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open sound path", command=selectimage)

Overframe = Frame(root, background="grey35")
Overframe.pack(fill="both", expand=True)

# Empty used for layout mangement
EmptyLabel1 = Label(Overframe, width=10, height=2, background="grey35")
EmptyLabel1.grid(row=0, column=4)
# EmptyLabel2 = Label(Overframe, width=3, height=2, background="grey35")
# EmptyLabel2.grid(row=0, column=0)
EmptyLabel3 = Label(Overframe, width=10, height=2, background="grey35")
EmptyLabel3.grid(row=0, column=2)

Filters = Frame(Overframe, pady=5, padx=5, background="grey35")
Filters.grid(row=0, column=1)

# Filter frame
FilterFrame = LabelFrame(Filters, text="Signal Processing Filters", padx=5, pady=5, font="Bold", height=350, width=450)
FilterFrame.pack_propagate(FALSE)
FilterFrame.pack(side=TOP)

# if column 0 set empty labels to column -1

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

# Image procsses frame
ProcessesFrame = LabelFrame(Filters, text="Image Proccessing Filters", pady=5, padx=5, font="Bold", height=350, width=450)
ProcessesFrame.pack_propagate(FALSE)
ProcessesFrame.pack(side=BOTTOM, pady=15)


BtnFrameProces = Frame(ProcessesFrame, pady=5)
BtnFrameProces.pack(side=BOTTOM)

Noise = Button(BtnFrameProces, text="Noise filter", command=createnoiseframe)
Noise.grid(row=0, column=0, padx=5)
Noise.config(state=DISABLED)

Sharp = Button(BtnFrameProces, text="Sharpening filter", command=createsharpframe)
Sharp.grid(row=0, column=1, padx=5)
Sharp.config(state=DISABLED)

Smooth = Button(BtnFrameProces, text="Smoothing filter", command=createsmoothframe)
Smooth.grid(row=0, column=2, padx=5)
Smooth.config(state=DISABLED)


# Preview frame
DisplayFrame = LabelFrame(Overframe, text="Preview", pady=5, padx=5)
DisplayFrame.grid(row=0, column=5)

UnderFrame = Frame(DisplayFrame)
UnderFrame.pack()

BtnFrame = Frame(UnderFrame, pady=5, padx=5)
BtnFrame.pack(side=BOTTOM)

ImageLabel = Label(UnderFrame, text="Your image will be displayed", width=42, height=24)
ImageLabel.pack(side=TOP)

PlayImage = Button(BtnFrame, text="Play sound", pady=5, padx=15, command=playmusic)
PlayImage.grid(row=0, column=0)
PlayImage.config(state=DISABLED)


FullImage = Button(BtnFrame, text="Show full image", pady=5, padx=5, command=getroot)
FullImage.grid(row=0, column=2)
FullImage.config(state=DISABLED)

# Middle display frame
FilteredImage = LabelFrame(Overframe, text="Image with choosen filter applied", width=600, height=500)
FilteredImage.grid(row=0, column=3)

FilterLabel = Label(FilteredImage, text="Your image will be displayed here", width=78, height=32)
FilterLabel.pack_propagate(FALSE)
FilterLabel.pack(side=TOP)

NumberFrame = Frame(FilteredImage, height=32)
NumberFrame.pack()

# Universal frame
UniversalFrame = Frame(FilteredImage, padx=5, pady=5)
UniversalFrame.pack(side=BOTTOM)


UniPlay = Button(UniversalFrame, text="Play Image", command=func_for_interface.playfilter)
UniPlay.pack()
UniPlay.config(state=DISABLED)

mainloop()