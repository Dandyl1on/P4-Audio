import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from scipy.io import wavfile
from PIL import Image, ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog, Label, Tk, messagebox as mb

import cv2

root = Tk()
root.title("App prototype")
root.geometry("500x500")

LoadImage = None
DisplayImage = None

def select():
    mb.showinfo("Choose", "Choose a picture you want to apply filters to")

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

    mb.showinfo("Sliders", "Use the sliders on the left to adjust the filter parameters")

def Apply():
    DisplayImage.destroy()

def SliderValues():
    print(Slider1.get())
    print(Slider2.get())
    print(Slider3.get())

def callback():
    if mb.askyesno("Verify", "Do you really want to Exit?"):
        mb.showwarning("Yes","Exit not yet implemented")
    else:
        mb.showinfo("No","Exit has been cancelled")

menu= Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open...", command=select)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=callback)

OverFrame = Frame(root, pady=5, padx=5, height=500)
OverFrame.pack()

SliderFrame = LabelFrame(OverFrame,text="Adjust these sliders", pady=50, padx=5, height=300)
SliderFrame.grid(row=0, column=0)

Slider1=Scale(SliderFrame, from_=0, to=200, orient=HORIZONTAL, troughcolor="darkgreen", background="green")
Slider1.grid(row=0, column=0)
Slider2=Scale(SliderFrame, from_=0, to=100, orient=HORIZONTAL, troughcolor="darkblue", background="blue")
Slider2.grid(row=1, column=0)
Slider3=Scale(SliderFrame, from_=0, to=50, tickinterval=15, orient=HORIZONTAL, troughcolor="darkred", background="red")
Slider3.grid(row=2, column=0)

Btn = Button(SliderFrame, text="Apply filter", command=lambda: [Apply(), SliderValues()])
Btn.grid(row=0, column=1)

ImageFrame = LabelFrame(OverFrame, text="This is the image", padx=5, pady=5)
ImageFrame.grid(row=0, column=1)

mainloop()