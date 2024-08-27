import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import PIL.Image
import cv2
import pygame
import time
import Equalloudness_transformation
from Equalloudness_transformation import *
import soundfile as sf

import all_filters
from all_filters import ApplyFilters

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
LowpassFrame = None
HighpassFrame = None
NotchFrame = None

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
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file",
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
    pygame.mixer.music.load('reconstructed_audio_from_combined_image.wav')
    pygame.mixer.music.play(loops=0)
    progressbar['value'] = 0
    proglabel.config(text="Waiting for sound clip")

def stop():
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

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
    HighBtn.config(state=NORMAL)
    LowBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)

def bandimage():
    global LoadImage
    global FImage
    global FLoad
    global CV2Image
    global sr

    all_filters.high_freq = BandHighSlider.get()
    all_filters.low_freq = BandLowSlider.get()

    ApplyFilters()

    CV2Image = cv2.imread("filtered_combined_image_bandpass.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
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
    Placement +=1

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
    Label1 = Label(BandpassFrame, text="Adjust highpass")
    Label1.pack()

    BandLowSlider = Scale(BandpassFrame, from_=0, to=22050, orient=HORIZONTAL, length=200)
    BandLowSlider.pack(padx=5, pady=5)
    Label2 = Label(BandpassFrame, text="Adjust lowpass")
    Label2.pack()

    Apply = Button(BandpassFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    BandBtn.config(state=DISABLED)
    HighBtn.config(state=NORMAL)
    LowBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)

    L.destroy()

    if LowpassFrame or HighpassFrame or NotchFrame is not None:
        destroylowpass()
        destroyhighpass()
        destroynotch()

def highpass():
    global HighpassFrame
    global BandBtn
    global HighBtn
    global LowBtn
    global NotchBtn
    global HighSlider

    HighpassFrame = LabelFrame(FilterFrame, text="Highpass", font="bold")
    HighpassFrame.pack()

    HighSlider = Scale(HighpassFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    HighSlider.pack(padx=5, pady=5)
    Label1 = Label(HighpassFrame, text="Adjust highpass")
    Label1.pack()

    Apply = Button(HighpassFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    BandBtn.config(state=NORMAL)
    HighBtn.config(state=DISABLED)
    LowBtn.config(state=NORMAL)
    NotchBtn.config(state=NORMAL)
    L.destroy()


    if BandpassFrame or LowpassFrame or NotchFrame is not None:
        destroybandpass()
        destroylowpass()
        destroynotch()

def lowpass():
    global LowpassFrame
    global BandBtn
    global HighBtn
    global LowBtn
    global NotchBtn
    global LowSlider

    LowpassFrame = LabelFrame(FilterFrame, text="Lowpass", font="Bold")
    LowpassFrame.pack()

    LowSlider = Scale(LowpassFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    LowSlider.pack(padx=5, pady=5)
    Label1 = Label(LowpassFrame, text="Adjust lowpass")
    Label1.pack()

    Apply = Button(LowpassFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    BandBtn.config(state=NORMAL)
    HighBtn.config(state=NORMAL)
    LowBtn.config(state=DISABLED)
    NotchBtn.config(state=NORMAL)

    L.destroy()

    if BandpassFrame or HighpassFrame or NotchFrame is not None:
        destroybandpass()
        destroyhighpass()
        destroynotch()

def notch():
    global NotchFrame
    global NotchSlider

    NotchFrame = LabelFrame(FilterFrame, text="Notch", font="bold")
    NotchFrame.pack()

    NotchSlider = Scale(NotchFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    NotchSlider.pack(padx=5, pady=5)
    Label1 = Label(NotchFrame, text="Adjust Notch")
    Label1.pack()

    Apply = Button(NotchFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    BandBtn.config(state=NORMAL)
    HighBtn.config(state=NORMAL)
    LowBtn.config(state=NORMAL)
    NotchBtn.config(state=DISABLED)

    L.destroy()

    if BandpassFrame or LowpassFrame or HighpassFrame is not None:
        destroybandpass()
        destroylowpass()
        destroyhighpass()

def destroybandpass():
    global BandpassFrame
    if BandpassFrame is not None:
        BandpassFrame.destroy()
        BandpassFrame = None

def destroyhighpass():
    global HighpassFrame
    if HighpassFrame is not None:
        HighpassFrame.destroy()
        HighpassFrame = None

def destroylowpass():
    global LowpassFrame
    if LowpassFrame is not None:
        LowpassFrame.destroy()
        LowpassFrame = None

def destroynotch():
    global NotchFrame
    if NotchFrame is not None:
        NotchFrame.destroy()
        NotchFrame = None

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
HighBtn = Button(BtnFrame, text="Highpass Filter", command=highpass)
HighBtn.grid(row=0, column=1, padx=5)
HighBtn.config(state=DISABLED)
LowBtn = Button(BtnFrame, text="Lowpass Filter", command=lowpass)
LowBtn.grid(row=0, column=2, padx=5)
LowBtn.config(state=DISABLED)
NotchBtn = Button(BtnFrame, text="Notch Filter", command=notch)
NotchBtn.grid(row=0, column=3, padx=5)
NotchBtn.config(state=DISABLED)

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