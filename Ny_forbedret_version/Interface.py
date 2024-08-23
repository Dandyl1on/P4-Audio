import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import PIL.Image
import cv2
import pygame
import time
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
Bandbtn = None


# Creates a sound player from pygame
pygame.mixer.init()

def select():
    global Filepath
    global progressbar
    global proglabel

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

    while progressbar['value'] < 100:
        progressbar['value'] += 20
        proglabel.config(text="Loading")
        root.update_idletasks()
        time.sleep(0.1)

    proglabel.config(text="Loading complete!")

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

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    Resize = CV2Image.resize((300,300), PIL.Image.LANCZOS)
    LoadImage = ImageTk.PhotoImage(Resize)

    if PlaceImage is None:
        PlaceImage = Label(DisplayFrame, image=LoadImage)
        PlaceImage.pack()
    else:
        PlaceImage.config(image=LoadImage)
    ImageLabel.destroy()


def bandimage():
    global LoadImage
    global FImage
    global FLoad

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    #Resize = CV2Image.resize((500, 500), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(CV2Image)

    if FImage is None:
        FImage = Label(FilteredImage, image=FLoad)
        FImage.pack()
    else:
        FImage.config(image=FLoad)
    FilterLabel.destroy()

# Filter Frames

def bandpass():
    global BandpassFrame
    global Bandbtn

    BandpassFrame = Frame(Overframe)
    BandpassFrame.grid(row=1, column=1)

    Title = Label(BandpassFrame, text="Bandpass filter", background="Light Green")
    Title.pack(pady=5)

    BandHighSlider = Scale(BandpassFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    BandHighSlider.pack(padx=5, pady=5)
    Label1 = Label(BandpassFrame, text="Adjust highpass")
    Label1.pack()

    BandLowSlider = Scale(BandpassFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    BandLowSlider.pack(padx=5, pady=5)
    Label2 = Label(BandpassFrame, text="Adjust lowpass")
    Label2.pack()

    Apply = Button(BandpassFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    if LowpassFrame or HighpassFrame or NotchFrame is not None:
        destroylowpass()
        destroyhighpass()
        destroynotch()

def highpass():
    global HighpassFrame

    HighpassFrame = Frame(Overframe)
    HighpassFrame.grid(row=1, column=1)

    Title = Label(HighpassFrame, text="Highpass Filter", background="Light Green")
    Title.pack(pady=5)

    HighSlider = Scale(HighpassFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    HighSlider.pack(padx=5, pady=5)
    Label1 = Label(HighpassFrame, text="Adjust highpass")
    Label1.pack()

    Apply = Button(HighpassFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    if BandpassFrame or LowpassFrame or NotchFrame is not None:
        destroybandpass()
        destroylowpass()
        destroynotch()

def lowpass():
    global LowpassFrame

    LowpassFrame = Frame(Overframe)
    LowpassFrame.grid(row=1, column=1)

    Title = Label(LowpassFrame, text="Lowpass filter", background="light green")
    Title.pack(pady=5)

    LowSlider = Scale(LowpassFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    LowSlider.pack(padx=5, pady=5)
    Label1 = Label(LowpassFrame, text="Adjust lowpass")
    Label1.pack()

    Apply = Button(LowpassFrame, text="Apply filter", command=bandimage)
    Apply.pack()

    if BandpassFrame or HighpassFrame or NotchFrame is not None:
        destroybandpass()
        destroyhighpass()
        destroynotch()

def notch():
    global NotchFrame

    NotchFrame = Frame(Overframe)
    NotchFrame.grid(row=1, column=1)

    Title = Label(NotchFrame, text="Notch filter", background="Light Green")
    Title.pack(pady=5)

    NotchSlider = Scale(NotchFrame, from_=0, to=10, orient=HORIZONTAL, length=200)
    NotchSlider.pack(padx=5, pady=5)
    Label1 = Label(NotchFrame, text="Adjust Notch")
    Label1.pack()

    Apply = Button(NotchFrame, text="Apply filter", command=bandimage)
    Apply.pack()

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

Overframe = Frame(root, background="black")
Overframe.pack(fill="both", expand=True)

#Empty
EmptyLabel= Label(Overframe, width=15, height=11, background="red")
EmptyLabel.grid(row=0, column=3)
EmptyLabel= Label(Overframe, width=55, height=10, background="blue")
EmptyLabel.grid(row=0, column=1)
EmptyLabel= Label(Overframe, width=20, height=10, background="green")
EmptyLabel.grid(row=0, column=0)
EmptyLabel= Label(Overframe, width=20, height=10, background="red")
EmptyLabel.grid(row=0, column=4)

# Entry frame
EntryFrame = LabelFrame(Overframe, text="Your sounds path directory")
EntryFrame.grid(row=0, column=2)

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
FilterFrame = LabelFrame(Overframe, text="Choose filters")
FilterFrame.grid(row=1, column=0)

BandBtn = Button(FilterFrame, text="Bandpass Filter", command=bandpass)
BandBtn.pack(pady=5, padx=5)
HighBtn = Button(FilterFrame, text="Highpass Filter", command=highpass)
HighBtn.pack(pady=5, padx=5)
LowBtn = Button(FilterFrame, text="Lowpass Filter", command=lowpass)
LowBtn.pack(pady=5, padx=5)
NotchBtn = Button(FilterFrame, text="Notch Filter", command=notch)
NotchBtn.pack(pady=5, padx=5)



# Display frame
DisplayFrame = LabelFrame(Overframe, text="Original image without filters")
DisplayFrame.grid(row=1, column=4)

ShowImage = Button(DisplayFrame, text="Show image", pady=5, padx=5, command=displayimage)
ShowImage.pack(side=TOP)

ImageLabel = Label(DisplayFrame, text="Your image will be displayed here", width=42, height=20)
ImageLabel.pack()

FilteredImage = LabelFrame(Overframe, text="Image with choosen filter applied") # use widht and height later
FilteredImage.grid(row=1, column=2)

FilterLabel = Label(FilteredImage, text="Your image will be displayed here")
FilterLabel.pack()

mainloop()