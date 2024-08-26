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
root.title("Name")
root.geometry("700x600")

menu = Menu(root)
root.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Open sound path")
fileMenu.add_command(label="Show full image")

root.config(background="grey37")

Name = Label(root, text="Name of program", background="grey37")
Name.pack()

OverFrame = Frame(root)
OverFrame.pack(expand=True, fill="both")

LeftSide = LabelFrame(OverFrame, text="LeftSide")
LeftSide.pack(side=LEFT)

MidSide = LabelFrame(OverFrame, text="MidSide")
MidSide.pack()

RightSide = LabelFrame(OverFrame, text="RightSide")
RightSide.pack(side=RIGHT)

test = Label(LeftSide, text="Test")
test.pack()
test = Label(MidSide, text="Test")
test.pack()
test = Label(RightSide, text="Test")
test.pack()

mainloop()