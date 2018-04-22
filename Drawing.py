from PIL import Image,ImageTk  
import tkinter as tk
from LoadAndSave import LoadImage
from ImageColor import LoadImageColor
from Plots import PlotGraph
import cv2
class GUI:
    CallBack=None
    ApplyOperation=None
    SaveImage=None
    PlotImage=None
    GetNames=None
    DropDown=None
    def __init__(self):
        self.root=tk.Tk()
        self.root.geometry("1250x700")
    def Run(self):
        self.root.mainloop()
    def InitializedComponents(self):
        tk.Label(self.root, text="Welcome to Computer Vision", font=('Comic Sans MS', 15)).grid(row=0)
        tk.Label(self.root, text="open the file ", font=('Comic Sans MS', 10)).grid(row=1)

        tk.Button(text='Open File', command=self.CallBack).grid(row=1,column=1,sticky=tk.W)
        
        tk.Button(text='Apply Operation',command=self.ApplyOperation).grid(row=2,column=0,sticky=tk.W)
        tk.Button(text='Save Image',command=self.SaveImage).grid(row=2,column=1,sticky=tk.W)
        tk.Button(text='Plot Image',command=self.PlotImage).grid(row=2,column=2,sticky=tk.W)
        
        mainframe = tk.Frame(self.root)
        mainframe.grid(column=3,row=1, sticky=(tk.N,tk.W,tk.E,tk.S) )
        mainframe.columnconfigure(0, weight = 1)
        mainframe.rowconfigure(0, weight = 1)
        self.DropDown = tk.StringVar(self.root)         
        choices = { 'Average Filter','Median Filter','Blur Filter','Gaussian Filter','Salt and Pepper Noise','Gaussian Noise'}
        self.DropDown.set('Average Filter') # set the default option         
        popupMenu = tk.OptionMenu(mainframe, self.DropDown, *choices)
        tk.Label(mainframe, text="Choose an Operation").grid(row = 1, column = 1)
        popupMenu.grid(row = 2, column =1)
        self.historyListBox = tk.Listbox(self.root)
        self.historyListBox.grid(row=2,column=7)
        tk.Button(text='Get Image',command=self.GetNames).grid(row=2,column=3,sticky=tk.W)


        
        
            