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
    mb.showinfo("Bandpass Information", "Adjust the low cut and high cut frequencies to pass only the values in between the values. \n\nHigh cut: Determines the high cutoff frequency every frequency higher than this value is not passed through. \n\nLow cut: Determines the low cutoff frequency every frequency lower than this value is not passed through. ")
def NotchInformation():
    mb.showinfo("Notch Information", "Adjust the frequency to determine the midpoint of the bandwidth. \n\nAdjust the bandwidth to determine the range of which the filter blocks frequencies. ")
def EqualizationInformation():
    mb.showinfo("Equalization Information", "Adjust the frequency bands to choose the frequency range of the 4 bands. (Note that the bands cannot overlap with one another). \n\nAdjust the equalization weight to determine the amount of increase/decrease you want. ")


def NoiseInformation():
    mb.showinfo("Noise information", "Adjust the mean to determine the average value of the noise added to the image. \n\nAdjust the sigma to determine the deviation of which the amount applied can vary.")
def SharpInformation():
    mb.showinfo("Sharpening Information", "Adjust the kernel size to choose how big the affected area for each modification should be.\n\n"
                                          "Alpha: Adjust the alpha to determine the weight of much of the original image is preserved in the final sharpening result. (A higher weight results in a more significant and sharper modification.\n"
                                          "Beta: Adjusts the amount of blur that is subtracted. (This can only be negative) \n"
                                          "Gamma: Adjusts a value which is added to each pixel after the sharpness is applied. (This is used to brighten the image.) \n"
                                          "Sigma: Adjusts the amount of blur applied to the the image. (A higher value results in a blurrier image.)")
def SmoothInformation():
    mb.showinfo("SmoothingInformation", "Adjust the kernel to choose the affected area.\n"
                                        "Adjust the weight to determine the amount of blending applied when smoothing. ")
