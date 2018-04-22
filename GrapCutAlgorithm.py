from GrapCutMouse import MouseEventHandler
import GrapCutSetting as setting
from tkinter.filedialog import askopenfilename
import tkinter as tk
import cv2
import numpy as np
def CallBack():  
        imgName= askopenfilename()
        return imgName
class GUI:
    def __init__(self):
        self.root=tk.Tk()
        self.root.geometry("200x200")
        self.output=0
        self.bar=0
        self.res=0
        self.mask2=0
        self.key=0
        self.imgName=None
        self.myObj=MouseEventHandler()
        self.InitializeComponents()
        self.root.mainloop()
    def InitializeComponents(self):
        tk.Label(self.root, text="Load your Image", font=('Comic Sans MS', 10)).grid(row=0)
        tk.Button(text='Open File', command=self.GrapCutAlgorithm).grid(row=0,column=1,sticky=tk.W)
        #tk.Button(text='Done', command=self.GrapCutAlgorithm).grid(row=1,column=1,sticky=tk.W)
    def GrapCutAlgorithm(self):
        self.imgName=CallBack()
        setting.img=cv2.imread(self.imgName)
        setting.img2=setting.img.copy()
        setting.mask=np.zeros(setting.img.shape[:2],dtype=np.uint8)
        self.output=np.zeros(setting.img.shape,np.uint8)
        cv2.namedWindow('output')
        cv2.namedWindow('input')
        cv2.setMouseCallback('input',self.myObj.onMouseClick)
        cv2.moveWindow('input',setting.img.shape[1]+10,90)
        print(" Draw a rectangle around the object using right mouse button \n")
        while(1):
            cv2.imshow('output',self.output)
            cv2.imshow('input',setting.img)
            self.key=cv2.waitKey(1)
            if self.key == 27: #Escape
                break
            elif self.key == ord('0'):
                print(" mark background regions with left mouse button \n")
                setting.value=setting.Draw_BackGround
            elif self.key == ord('1'):
                print(" mark foreground regions with left mouse button \n")
                setting.value=setting.Draw_ForeGround
            elif self.key == ord('2'):
                setting.value=setting.Draw_PR_BackGround
            elif self.key == ord('3'):
                setting.value=setting.Draw_PR_ForeGround
            elif self.key == ord('s'):
                self.bar=np.zeros((setting.img.shape[0],5,3),np.uint8)
                self.res=np.hstack((setting.img2,self.bar,setting.img,self.bar,self.output))
                tmp=cv2.cvtColor(self.output,cv2.COLOR_BGR2GRAY)
                _,alpha=cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                b,g,r=cv2.split(self.output)
                rgba=[b,g,r,alpha]
                dst=cv2.merge(rgba,4)
                cv2.imwrite("C:\\Users\\mohamed ismail\\Desktop\\Result2.png",dst)
                cv2.imwrite("C:\\Users\\mohamed ismail\\Desktop\\Result.png",self.res)
                print(" Result saved as image \n")
            elif self.key == ord('r'):
                setting.rect=(0,0,1,1)
                setting.drawing=False
                setting.rectangle=False
                setting.rect_or_mask=100
                setting.rect_over=False
                setting.value=setting.Draw_ForeGround
                setting.img=setting.img2.copy()
                setting.mask=np.zeros(setting.img.shape[:2],dtype=np.uint8)
                self.output=np.zeros(setting.img.shape,np.uint8)
                print("resetting \n")
            elif self.key == ord('n'):
                if (setting.rect_or_mask == 0):         # grabcut with rect
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    cv2.grabCut(setting.img2,setting.mask,setting.rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                    setting.rect_or_mask = 1
                elif setting.rect_or_mask == 1:         # grabcut with mask
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    cv2.grabCut(setting.img2,setting.mask,setting.rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
                print("For finer touchups, mark foreground and background after pressing keys 0-3 and again press n")
            self.mask2 = np.where((setting.mask==1) + (setting.mask==3),255,0).astype('uint8')
            self.output = cv2.bitwise_and(setting.img2,setting.img2,mask=self.mask2)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    myObj=GUI()
    myObj.InitializeComponents()