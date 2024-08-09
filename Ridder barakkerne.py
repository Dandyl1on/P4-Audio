import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from scipy.io import wavfile
from PIL import Image, ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog, Label, Tk

root = Tk()
root.title("App prototype")
root.geometry("500x500")

def select():
    root.filename = filedialog.askopenfilename(
        initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file",
        filetypes=(("PNG files", "*.png"),))
    myLabel = Label(imgFrame, text=root.filename).pack()
    myImage = ImageTk.PhotoImage(PIL.Image.open(root.filename))
    imageLabel = Label(imgFrame, image=myImage).pack()
    submit()
    clear()


def clear():
    DescriptText.delete(1.0,END)
def submit():
    imgFrame.config(text=DescriptText.get(1.0, END))


#A Grid frame for buttons
buttonframe= LabelFrame(root, text="Click these buttons i fucking dare you >:(", padx=5, pady=5, background="tomato")
buttonframe.pack()

#Select button for choosing a picture
Selectbtn=Button(buttonframe, text="Select picture", command=select, background="tomato")
Selectbtn.grid(row=0, column=2)

#Clear button for clearing any text
clear_button=Button(buttonframe, text="Clear text box", command=clear)
clear_button.grid(row=0, column=0)

#Submit button for setting the text into the label below grid frame for buttons
Submit=Button(buttonframe, text="Submit text", command=submit)
Submit.grid(row=0, column=1)

DescriptText = Text(root, width=50, height=2, background="chocolate3")
DescriptText.pack(pady=5)

imgFrame= LabelFrame(root, text="This is image frame", padx=5, pady=5)
imgFrame.pack(padx=10, pady=10)



mainloop()