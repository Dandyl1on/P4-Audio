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

LoadImage = None
BandIMG = None
PlaceImage = None
FImage = None
FLoad = None

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

def Bandpass():
    SliderFrame = LabelFrame(Overframe, text="Bandpass Filter")
    SliderFrame.grid(row=1, column=1)
    Slider1 = Scale(SliderFrame, from_=0, to=10, orient=HORIZONTAL)
    Slider1.pack()
    Slider2 = Scale(SliderFrame, from_=0, to=10, orient=HORIZONTAL)
    Slider2.pack()

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

BandBtn = Button(FilterFrame, text="Bandpass Filter", command=Bandpass)
BandBtn.pack(pady=5, padx=5)
NotchBtn = Button(FilterFrame, text="Notch Filter", command=bandimage)
NotchBtn.pack(pady=5, padx=5)
LowBtn = Button(FilterFrame, text="Lowpass Filter")
LowBtn.pack(pady=5, padx=5)
HighBtn = Button(FilterFrame, text="Highpass Filter")
HighBtn.pack(pady=5, padx=5)

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