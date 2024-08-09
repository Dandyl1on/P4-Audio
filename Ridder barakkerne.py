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

def clear():
    My_text.delete(1.0,END)
def submit():
    My_label.config(text=My_text.get(1.0, END))

imgFrame= LabelFrame(root, text="This is image frame", padx=5, pady=5)
imgFrame.pack(padx=10, pady=10)

My_text = Text(root, width=50, height=2)
My_text.pack(pady=5)

buttonframe= Frame(root)
buttonframe.pack()

clear_button=Button(buttonframe, text="Clear screen", command=clear)
clear_button.grid(row=0, column=0)

Submit=Button(buttonframe, text="Submit text", command=submit)
Submit.grid(row=0, column=1)

My_label= Label(root, text='')
My_label.pack(pady=5)

root.filename = filedialog.askopenfilename(initialdir="C:/Users/marku/OneDrive - Aalborg Universitet/Githubs/P4-Audio", title="select a file", filetypes=(("PNG files", "*.png"),))
myLabel = Label(imgFrame, text=root.filename).pack()
myImage = ImageTk.PhotoImage(PIL.Image.open(root.filename))
imageLabel= Label(imgFrame, image=myImage).pack()


mainloop()
