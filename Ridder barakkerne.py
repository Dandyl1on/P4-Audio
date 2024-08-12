import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from scipy.io import wavfile
from PIL import Image, ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog, Label, Tk
import cv2

root = Tk()
root.title("App prototype")
root.geometry("500x500")

LoadImage = None

def select():
    global LoadImage
    global DisplayImage
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file",
        filetypes=(("PNG files", "*.png"), ("All files", "*"))
    )
    CV2Image = cv2.imread(root.filename, cv2.IMREAD_UNCHANGED)
    if CV2Image.dtype == np.uint16:
        CV2Image = (CV2Image / 256).astype('uint8')
    CV2Image = cv2.cvtColor(CV2Image, cv2.COLOR_BGR2RGB)
    CV2Image = PIL.Image.fromarray(CV2Image)
    LoadImage = ImageTk.PhotoImage(CV2Image)
    DisplayImage = Label(ImageFrame, image=LoadImage)
    DisplayImage.pack()

def Apply():
    #global LoadImage
    #LoadImage = None
    DisplayImage.destroy()

OverFrame = Frame(root, pady=5, padx=5)
OverFrame.pack()

SliderFrame = LabelFrame(OverFrame,text="Adjust these sliders", pady=5, padx=5)
SliderFrame.grid(row=0, column=0)

Slider1=Scale(SliderFrame, from_=0, to=200, orient=HORIZONTAL, troughcolor="darkgreen", background="green")
Slider1.grid(row=0, column=0)
Slider1=Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL, troughcolor="darkblue", background="blue")
Slider1.grid(row=1, column=0)
Slider1=Scale(SliderFrame, from_=0, to=50, tickinterval=15, orient=HORIZONTAL, troughcolor="darkred", background="red")
Slider1.grid(row=2, column=0)

Btn = Button(SliderFrame, text="Apply filter", command=Apply)
Btn.grid(row=0, column=1)

ImageFrame = LabelFrame(OverFrame, text="This is the image", padx=5, pady=5)
ImageFrame.grid(row=0, column=1)

Btn2 = Button(ImageFrame, text="Test", command=select)
Btn2.pack()



mainloop()