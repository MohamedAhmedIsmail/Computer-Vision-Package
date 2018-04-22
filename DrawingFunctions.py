import tkinter as tk
from tkinter.filedialog import askopenfilename
from imageInTk import ReadImageInGui  
from Filters import FilterImage
from Plots import PlotGraph
from LoadAndSave import LoadImage
from ImageColor import LoadImageColor
class DrawingHelperFunction:
    
    imagePlotter = ReadImageInGui()
    def __init__(self,gui):
        self.gui=gui
        self.checkedfiledialog=tk.IntVar()
        self.dropdown=None
        self.savedImage=None
        self.filterImage=None
        self.noiseImage=None
        self.myList=[]
    def Bind(self):
        self.gui.CallBack=self.callBack
        self.gui.ApplyOperation=self.applyOperations
        self.gui.SaveImage=self.SaveImage
        self.gui.PlotImage=self.PlotMyFilteredImage
        self.gui.GetNames=self.getNames
        self.gui.InitializedComponents()
        self.dropdown = self.gui.DropDown
        
    def callBack(self):
        #print("Hello There 1")
        if self.checkedfiledialog.get() !=True:    
            imgName= askopenfilename()
            self.imagePlotter.PlotImageInGUI(imgName)
            image=LoadImage.LoadMyImage(imgName,LoadImageColor.color)
            self.savedImage= image
            print(imgName)
    
    def OutPutPath(self):
        
        file=tk.filedialog.asksaveasfilename()
        return file
   
        
    def applyOperations(self):
        #print("Hello There 2")
        if self.dropdown.get()=='Average Filter':
            filteredImage=FilterImage.AverageFilter(self.savedImage)
            self.filterImage=filteredImage
            self.imagePlotter.PlotMyNewImage(self.filterImage)
            return filteredImage
            
        elif self.dropdown.get()=='Median Filter':
            filteredImage=FilterImage.MedianFilter(self.savedImage)
            self.filterImage=filteredImage
            self.imagePlotter.PlotMyNewImage(self.filterImage)
            return filteredImage
            
        elif self.dropdown.get()=='Blur Filter':
            filteredImage=FilterImage.BlurFilter(self.savedImage)
            self.filterImage=filteredImage
            self.imagePlotter.PlotMyNewImage(self.filterImage)
            return filteredImage
        
        elif self.dropdown.get()=='Gaussian Filter':
            filteredImage=FilterImage.GaussianFilter(self.savedImage)
            self.filterImage=filteredImage
            self.imagePlotter.PlotMyNewImage(self.filterImage)
            return filteredImage
        elif self.dropdown.get()=='Salt and Pepper Noise':
            noisyImage=FilterImage.SaltAndPepper(self.savedImage)
            self.noiseImage=noisyImage
            self.imagePlotter.PlotMyNewImage(self.noiseImage)
            return noisyImage
            
        elif self.dropdown.get()=='Gaussian Noise':
            noisyImage=FilterImage.GaussianNoise(self.savedImage)
            self.noiseImage=noisyImage
            self.imagePlotter.PlotMyNewImage(self.noiseImage)
            return noisyImage
        return None
       
    def SavedImage(self):
        #print("Hello There 3")
        myOutPath=self.OutPutPath()
        #print(myOutPath)
        myImage=self.applyOperations()
        LoadImage.SaveMyImage(myOutPath,myImage)
        self.myList.append(myOutPath)
        return myOutPath
        
    def PlotMyFilteredImage(self):
        if self.filterImage!=None:
            PlotGraph.Plot2Images(self.savedImage,self.filterImage)
        else:
            PlotGraph.Plot2Images(self.savedImage,self.noiseImage)
    def PlotImage(self):
        if self.filterImage!=None:
            PlotGraph.PlotMyImage(self.applyOperations())
        else:
            PlotGraph.PlotMyImage(self.applyOperations())
    def SaveImage(self):
        savedPath = self.SavedImage()
        self.gui.historyListBox.insert(tk.END, savedPath)
    def getNames(self):
        myPathIndex =self.gui.historyListBox.curselection()[0]
        myPath = self.gui.historyListBox.get(myPathIndex)
        self.PlotImage()
        print(myPath)
