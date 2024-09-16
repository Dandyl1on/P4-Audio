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

import sharpening_filter
from sharpening_filter import *

import Smoothing
from Smoothing import *

import Noise
from Noise import *

NumberPlacement = 0

def displayimage():
    global LoadImage
    global PlaceImage
    global File
    global CV2Image
    global Start
    global StartImage
    global Displayed
    global CV2Image3
    global width
    global noresize

    CV2Image = cv2.imread("combined_image.png", cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image2 = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image3 = PIL.Image.fromarray(CV2Image2)
    height, width = CV2Image.shape[:2]

    noresize = ImageTk.PhotoImage(CV2Image3)
    # Redundant?
    Displayedresize = CV2Image3.resize((400,400), PIL.Image.LANCZOS)
    Displayed = ImageTk.PhotoImage(Displayedresize)

    Resize = CV2Image3.resize((300, 300), PIL.Image.LANCZOS)
    LoadImage = ImageTk.PhotoImage(Resize)

    return LoadImage, Displayed, width, noresize

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

def equalimage(bass, mid, uppermid, high, range1, range2, range3, range4):
    global FImage
    global FLoad
    global CV2Image

    stop()

    Equalization.low = bass
    Equalization.mid = mid
    Equalization.upper = uppermid
    Equalization.high = high
    Equalization.range1 = range1
    Equalization.range2 = range2
    Equalization.range3 = range3
    Equalization.range4 = range4

    Equalization.mainfunc()

    CV2Image = cv2.imread("equalized_image.png", cv2.IMREAD_UNCHANGED)
    CV2Image = PIL.Image.fromarray(CV2Image)
    CV2Image.save("Equal Image.png")
    Equalloudness_transformation.Path = "Equal Image.png" # WHAT(´･ω･`)?
    Resize = CV2Image.resize((550, 490), PIL.Image.LANCZOS)
    FLoad = ImageTk.PhotoImage(Resize)

    Equalloudness_transformation.infoFunc()

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

def Sharppathchange(Kernelval, Sigmaval, Aplhaval, Betaval, Gammaval):
    global SharpImageFinal
    stop()

    sharpfunction.image = CV2Image
    sharpfunction(Kernelval, Sigmaval, Aplhaval, Betaval, Gammaval)

    SharpImage = cv2.imread("Sharpening.png", cv2.IMREAD_UNCHANGED)
    if SharpImage.dtype == np.uint16:
        SharpImage = (CV2Image / 256).astype('uint8')
    SharpImage = cv2.cvtColor(SharpImage, cv2.COLOR_BGR2RGB)
    SharpImage = PIL.Image.fromarray(SharpImage)

    Equalloudness_transformation.Path = "Sharpening.png"
    Equalloudness_transformation.infoFunc()

    SharpImageFinal = ImageTk.PhotoImage(SharpImage)

    return SharpImageFinal

def Smoothpathchange(Kernelval, Weigthval):
    global SmoothImageFinal
    stop()

    Smoothing.image = CV2Image
    Smoothing.apply_smoothing(Kernelval, Weigthval)

    SmoothImage = cv2.imread("Smoothing.png", cv2.IMREAD_UNCHANGED)
    if SmoothImage.dtype == np.uint16:
        SmoothImage = (CV2Image / 256).astype('uint8')
    SmoothImage = cv2.cvtColor(SmoothImage, cv2.COLOR_BGR2RGB)
    SmoothImage = PIL.Image.fromarray(SmoothImage)

    Equalloudness_transformation.Path = "Smoothing.png"
    Equalloudness_transformation.infoFunc()

    SmoothImageFinal = ImageTk.PhotoImage(SmoothImage)

    return SmoothImageFinal

def Noisepathchange(Meanval, Sigmaval):
    global NoiseImageFinal
    stop()

    Noise.image = CV2Image
    Noise.add_gaussian_noise(Meanval, Sigmaval)

    NoiseImage = cv2.imread("Noise.png", cv2.IMREAD_UNCHANGED)
    if NoiseImage.dtype == np.uint16:
        NoiseImage = (CV2Image / 256).astype('uint8')
    NoiseImage = cv2.cvtColor(NoiseImage, cv2.COLOR_BGR2RGB)
    NoiseImage = PIL.Image.fromarray(NoiseImage)

    Equalloudness_transformation.Path = "Noise.png"
    Equalloudness_transformation.infoFunc()

    NoiseImageFinal = ImageTk.PhotoImage(NoiseImage)

    return NoiseImageFinal

def BandInformation():
    mb.showinfo("Bandpass Information", "Adjust the low cutoff and high cutoff to keep anything inbetween those two values")
def NotchInformation():
    mb.showinfo("Notch Information", "Adjust the Bandwidth to remove anything inside. The notch decides where on the image the bandwidth is placed")
def EqualizationInformation():
    mb.showinfo("Equalization Information", "Bands cannot be higher than the previous, bands decide how much of the image is modified by the frequencies. The frequencies decide how much the selected bands are modified")


def NoiseInformation():
    mb.showinfo("Noise information", "Adjust the mean for the noise, and the weight for the amount")
def SharpInformation():
    mb.showinfo("Sharpening Information", "Adjust the kernel size and the Sigma for the gaussian blur. The alhpa, beta and gamma is for the weight ")
def SmoothInformation():
    mb.showinfo("SmoothingInformation", "Adjust the kernel size for gaussian blur and the weight values")
