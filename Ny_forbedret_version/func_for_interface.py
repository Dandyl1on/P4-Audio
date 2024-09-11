import pygame
import cv2

import PIL.Image
from PIL.ImageFilter import Kernel
from scipy.io import wavfile
from PIL import Image, ImageTk
from fileinput import filename
from tkinter import *
from tkinter import filedialog, Label, Tk, messagebox as mb, ttk

import all_filters
from all_filters import ApplyFilters

import Equalloudness_transformation
from Equalloudness_transformation import *

import Equalization
from Equalization import *

NumberPlacement = 0

def displayimage():
    global LoadImage
    global PlaceImage
    global File
    global CV2Image
    global Start
    global StartImage
    global Displayed

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    Displayed = ImageTk.PhotoImage(CV2Image)
    Resize = CV2Image.resize((300, 300), PIL.Image.LANCZOS)
    LoadImage = ImageTk.PhotoImage(Resize)

    return LoadImage, Displayed

# def PlacingNumbers():
#     global NumberPlacement
#     for i in range(23):
#         NumberLabel = Label(NumberFrame, text=NumberPlacement, width=1, height=1)
#         NumberLabel.grid(row=NumberPlacement)
#
#         NumberPlacement += 1

def play(file):
    # Plays the sound in the load method
    pygame.mixer.music.load(file)
    pygame.mixer.music.play(loops=0)

def playfilter():
    pygame.mixer.music.unload()
    pygame.mixer.music.load('reconstructed_audio_from_combined_image.wav')
    pygame.mixer.music.play(loops=0)

def stop():
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    print("Music playing:", pygame.mixer.music.get_busy())


def bandimage(high, low):
    global FImage
    global FLoad
    global CV2Image
    # global FilterLabel

    stop()
    print(high, low)
    all_filters.high_freq = high
    all_filters.low_freq = low

    ApplyFilters()

    CV2Image = cv2.imread("filtered_combined_image_bandpass.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Bandpass Image.png")

    Equalloudness_transformation.Path = "Bandpass Image.png"

    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    Equalloudness_transformation.infoFunc()

    return FLoad

def notchimage(notch, bandwidth):
    global FImage
    global FLoad
    global CV2Image

    stop()

    all_filters.notch_freq = notch
    all_filters.bandwidth = bandwidth

    ApplyFilters()

    CV2Image = cv2.imread("filtered_combined_image_notch.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Notch Image.png")
    Equalloudness_transformation.Path = "Notch Image.png"
    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    Equalloudness_transformation.infoFunc()

    return FLoad

def equalimage(bass, mid, uppermid, high):
    global FImage
    global FLoad
    global CV2Image

    stop()

    Equalization.low = bass
    Equalization.mid = mid
    Equalization.upper = uppermid
    Equalization.high = high

    Equalization.mainfunc()

    CV2Image = cv2.imread("equalized_image.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Equal Image.png")
    Equalloudness_transformation.Path = "Equal Image.png" # WHAT(´･ω･`)?
    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    return FLoad

def fullimage(root):
    global CV2Image
    global Full

    LargeImage = Toplevel(root)
    LargeImage.title("Full Image")
    LargeImage.geometry("750x680")

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)

    Full = ImageTk.PhotoImage(CV2Image)
    NewImage = Label(LargeImage, image=Full)
    NewImage.pack(pady=10, padx=10)

