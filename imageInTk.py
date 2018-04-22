import tkinter as tk
from PIL import Image
from PIL import ImageTk
import cv2
class ReadImageInGui:
    
    def PlotImageInGUI(self,path=None):
        panelA=None
        panelB=None
        if len(path)>0:
            image =cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            
        if panelA is None or panelB is None:
            panelA = tk.Label(image=image)
            panelA.image = image
            panelA.grid(row=4,column=0)
            panelB =tk.Label(image=image)
            panelB.image=image
            panelB.grid(row=4,column=2)
        else:
            panelA.configure(image=image)
            panelA.image = image
            panelB.configure(image=image)
            panelB.image=image
            
    def PlotMyNewImage(self,image=None):
        panelA=None
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        if panelA is None:
            panelA=tk.Label(image=image)
            panelA.image = image
            panelA.grid(row=4,column=2)
        else:
            panelA.configure(image=image)
            panelA.image = image