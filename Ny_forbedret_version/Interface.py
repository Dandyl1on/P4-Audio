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

Placement = 0
photo_image_references = []

root = Tk()
root.title("Modifun")
root.geometry("700x600")

File = None
CV2Image = None
sr = None
SImage = None
SmallImageLoad = None
HighBtn = None
LowBtn = None
BandwidthSlider = None
BassSlider = None
MidSlider = None
UppermidSlider = None
HigherSlider = None

BandLowSlider = None
BandHighSlider = None
NotchSlider = None

# None types for Image displaying
LoadImage = None
BandIMG = None
PlaceImage = None
FImage = None
FLoad = None

# None types for filter sliders
BandpassFrame = None
NotchFrame = None
EqualFrame = None

# Creates a sound player from pygame
pygame.mixer.init()

def select():
    global Filepath
    global progressbar
    global proglabel
    global File

    stop()

    # The filedialog.askopenfilename ask the user to choose a .png or all files to open in the program
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio/Ny_forbedret_version", title="select a file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*"))
    )
    Filepath.delete(0, END)
    Filepath.insert(0, root.filename)

    File = os.path.basename(root.filename)

    Equalloudness_transformation.audio_path = root.filename

    print(Equalloudness_transformation.audio_path)
    Equalloudness_transformation.mainfunc()

    while progressbar['value'] < 100:
        progressbar['value'] += 20
        proglabel.config(text="Loading")
        root.update_idletasks()
        time.sleep(0.1)

    proglabel.config(text="Loading complete!")
    ImageLabel.config(text=File)
    ImageLabel.config(height=0)
    displayimage()

def play():
    # Plays the sound in the load method
    pygame.mixer.music.load(File)
    pygame.mixer.music.play(loops=0)
    progressbar['value'] = 0
    proglabel.config(text="Waiting for sound clip")

def playfilter():
    pygame.mixer.music.unload()
    pygame.mixer.music.load('reconstructed_audio_from_combined_image.wav')
    pygame.mixer.music.play(loops=0)
    progressbar['value'] = 0
    proglabel.config(text="Waiting for sound clip")

def stop():
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    print("Music playing:", pygame.mixer.music.get_busy())


def displayimage():
    global LoadImage
    global PlaceImage
    global File
    global CV2Image

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    Resize = CV2Image.resize((300, 300), PIL.Image.LANCZOS)
    LoadImage = ImageTk.PhotoImage(Resize)

    if PlaceImage is None:
        PlaceImage = Label(UnderFrame, image=LoadImage)
        PlaceImage.pack()
    else:
        PlaceImage.config(image=LoadImage)

    PlayImage.config(state=NORMAL)
    FullImage.config(state=NORMAL)
    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=NORMAL)

def bandimage():
    global LoadImage
    global FImage
    global FLoad
    global CV2Image
    global sr

    stop()

    all_filters.high_freq = BandHighSlider.get()
    all_filters.low_freq = BandLowSlider.get()

    ApplyFilters()

    CV2Image = cv2.imread("filtered_combined_image_bandpass.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Bandpass Image.png")
    Equalloudness_transformation.Path = "Bandpass Image.png"
    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    Equalloudness_transformation.mainfunc()

    if FImage is None:
        FImage = Label(FilteredImage, image=FLoad)
        FImage.pack()
    else:
        FImage.config(image=FLoad)
    FilterLabel.destroy()

def notchimage():
    global LoadImage
    global FImage
    global FLoad
    global CV2Image
    global sr

    stop()

    all_filters.notch_freq = NotchSlider.get()
    all_filters.bandwidth = BandwidthSlider.get()

    ApplyFilters()

    CV2Image = cv2.imread("filtered_combined_image_notch.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Notch Image.png")
    Equalloudness_transformation.Path = "Notch Image.png"
    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    Equalloudness_transformation.mainfunc()

    if FImage is None:
        FImage = Label(FilteredImage, image=FLoad)
        FImage.pack()
    else:
        FImage.config(image=FLoad)
    FilterLabel.destroy()


def equalimage():
    global LoadImage
    global FImage
    global FLoad
    global CV2Image
    global sr

    stop()

    Equalization.low = BassSlider.get()
    Equalization.mid = MidSlider.get()
    Equalization.upper = UppermidSlider.get()
    Equalization.high = HigherSlider.get()

    Equalization.mainfunc()

    CV2Image = cv2.imread("equalized_image.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Equal Image.png")
    Equalloudness_transformation.Path = "Equal Image.png"
    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    if FImage is None:
        FImage = Label(FilteredImage, image=FLoad)
        FImage.pack()
    else:
        FImage.config(image=FLoad)
    FilterLabel.destroy()


def getsr():
    return sr

def fullimage():
    global CV2Image
    global FLoad

    LargeImage = Toplevel(root)
    LargeImage.title("Full Image")
    LargeImage.geometry("750x680")

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)

    FLoad = ImageTk.PhotoImage(CV2Image)
    NewImage = Label(LargeImage, image=FLoad)
    NewImage.pack(pady=10, padx=10)

def saveimage():
    global LoadImage
    global CV2Image
    global SImage
    global SmallImageLoad
    global File
    global Placement

    SImage = Label(scrollframe)
    SImage.grid(row=Placement, column=0)

    text = Label(scrollframe, text=File, padx=10, pady=5, width=15)
    text.grid(row=Placement, column=1)

    Size = CV2Image.resize((50, 50), PIL.Image.LANCZOS)
    SmallImageLoad = ImageTk.PhotoImage(Size)

    photo_image_references.append(SmallImageLoad)

    SImage.config(image=SmallImageLoad)
    Placement += 1

def bandpass():
    global BandpassFrame
    global BandBtn
    global HighBtn
    global LowBtn
    global NotchBtn
    global BandHighSlider
    global BandLowSlider

    BandpassFrame = LabelFrame(FilterFrame, text="Bandpass", font="BOLD")
    BandpassFrame.pack()

    BandHighSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200)
    BandHighSlider.pack(padx=5, pady=5)
    Label1 = Label(BandpassFrame, text="Adjust highcut frequency")
    Label1.pack()

    BandLowSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200)
    BandLowSlider.pack(padx=5, pady=5)
    Label2 = Label(BandpassFrame, text="Adjust lowcut frequency")
    Label2.pack()

    Apply = Button(BandpassFrame, text="Apply filter", command=bandimage)
    Apply.pack(side=LEFT)
    Apply = Button(BandpassFrame, text="Play sound", command=playfilter)
    Apply.pack(side=RIGHT)

    BandBtn.config(state=DISABLED)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=NORMAL)

    L.destroy()

    if EqualFrame or NotchFrame is not None:
        destroynotch()
        destroyequal()

def notch():
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

    Apply = Button(NotchFrame, text="Apply filter", command=notchimage)
    Apply.pack(side=LEFT)
    Apply = Button(NotchFrame, text="Play sound", command=playfilter)
    Apply.pack(side=RIGHT)

    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=DISABLED)
    EqualBtn.config(state=NORMAL)

    L.destroy()

    if EqualFrame or BandpassFrame is not None:
        destroybandpass()
        destroyequal()

def equalization():
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

    Apply = Button(EqualFrame, text="Apply filter", command=equalimage)
    Apply.pack(side=LEFT)
    Apply = Button(EqualFrame, text="Play sound", command=playfilter)
    Apply.pack(side=RIGHT)

    BandBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    EqualBtn.config(state=DISABLED)

    L.destroy()

    if BandpassFrame or NotchFrame is not None:
        destroybandpass()
        destroynotch()

def destroybandpass():
    global BandpassFrame
    if BandpassFrame is not None:
        BandpassFrame.destroy()
        BandpassFrame = None

def destroynotch():
    global NotchFrame
    if NotchFrame is not None:
        NotchFrame.destroy()
        NotchFrame = None

def destroyequal():
    global EqualFrame
    if EqualFrame is not None:
        EqualFrame.destroy()
        EqualFrame = None

# Creates a menu in the interface, where the select function is called
menu = Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open sound path", command=select)
fileMenu.add_command(label="Show full image", command=fullimage)

Overframe = Frame(root, background="grey35")
Overframe.pack(fill="both", expand=True)

# Empty used for layout mangement
EmptyLabel = Label(Overframe, width=10, height=11, background="grey35")
EmptyLabel.grid(row=0, column=4)
EmptyLabel = Label(Overframe, width=3, height=10, background="grey35")
EmptyLabel.grid(row=0, column=0)
EmptyLabel = Label(Overframe, width=9, height=11, background="grey35")
EmptyLabel.grid(row=0, column=2)

# Entry frame
EntryFrame = LabelFrame(Overframe, text="Your sounds path directory")
EntryFrame.grid(row=0, column=3)

Playsound = Button(EntryFrame, text="Play sound", command=play)
Playsound.pack(pady=5, padx=5)

Filepath = Entry(EntryFrame, width=80)
Filepath.pack(pady=10)

progressbar = ttk.Progressbar(EntryFrame, orient=HORIZONTAL)
progressbar.pack(pady=5, padx=5)
progressbar['value'] = 0
proglabel = Label(EntryFrame, text="Waiting for sound clip")
proglabel.pack(pady=5, padx=5)

# Filter frame
FilterFrame = LabelFrame(Overframe, text="Choose filters", padx=5, pady=5)
FilterFrame.grid(row=1, column=1)

L = Label(FilterFrame, text="", width=31, height=10)
L.pack(side=TOP)

BtnFrame = Frame(FilterFrame, pady=5)
BtnFrame.pack(side=BOTTOM)

BandBtn = Button(BtnFrame, text="Bandpass Filter", command=bandpass)
BandBtn.grid(row=0, column=0, padx=5)
BandBtn.config(state=DISABLED)
NotchBtn = Button(BtnFrame, text="Notch Filter", command=notch)
NotchBtn.grid(row=0, column=1, padx=5)
NotchBtn.config(state=DISABLED)
EqualBtn = Button(BtnFrame, text="Equalization Filter", command=equalization)
EqualBtn.grid(row=0, column=2, padx=5)
EqualBtn.config(state=DISABLED)

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

PlayImage = Button(BtnFrame, text="Play sound", pady=5, padx=15, command=play)
PlayImage.grid(row=0, column=0)
PlayImage.config(state=DISABLED)

SaveImage = Button(BtnFrame, text="Save Image", padx=5, pady=5, command=saveimage)
SaveImage.grid(row=0, column=1)

# SaveImage.config(state=DISABLED)

FullImage = Button(BtnFrame, text="Show full image", pady=5, padx=5, command=fullimage)
FullImage.grid(row=0, column=2)
FullImage.config(state=DISABLED)


# Small Images
SmallImages = LabelFrame(F, text="Instances")
SmallImages.grid(row=2, column=4)

canvas = Canvas(SmallImages, height=70, width=260)
canvas.grid(row=0, column=0, sticky="nsew")

scroll = Scrollbar(SmallImages, command=canvas.yview)
scroll.grid(row=0, column=1, sticky="ns")

scrollframe = Frame(canvas, pady=5, padx=5)

canvas.create_window((0, 0), window=scrollframe, anchor="nw")
canvas.configure(yscrollcommand=scroll.set)

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollframe.bind("<Configure>", on_frame_configure)

SImage2 = Label(scrollframe, text="", padx=10, pady=5, width=15)
SImage2.grid(row=1, column=0)

text2 = Label(scrollframe, text="", padx=10, pady=5, width=15)
text2.grid(row=1, column=1)

SImage3 = Label(scrollframe, text="", padx=10, pady=5, width=15)
SImage3.grid(row=2, column=0)

text3 = Label(scrollframe, text="", padx=10, pady=5, width=15)
text3.grid(row=2, column=1)

SImage4 = Label(scrollframe, text="peekaboo", padx=10, pady=5, width=15)
SImage4.grid(row=3, column=0)

text4 = Label(scrollframe, text="", padx=10, pady=5, width=15)
text4.grid(row=3, column=1)

SmallImages.grid_rowconfigure(0, weight=1)
SmallImages.grid_columnconfigure(0, weight=1)

# Middle display frame
FilteredImage = LabelFrame(Overframe, text="Image with choosen filter applied", width=600, height=500)
FilteredImage.grid(row=1, column=3)

FilterLabel = Label(FilteredImage, text="Your image will be displayed here", width=78, height=32)
FilterLabel.pack()

mainloop()