import cv2
from ImageColor import LoadImageColor
class LoadImage:
    def __init__(self,imgPath,loadImageColor):
        self.imgPath=imgPath
        self.loadImageColor=loadImageColor
    @staticmethod
    def LoadMyImage(imgPath,loadImageColor):
        img=cv2.imread(imgPath)
        if loadImageColor == LoadImageColor.color:
            myImg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif loadImageColor == LoadImageColor.grayScale:
            myImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return myImg
    @staticmethod
    def SaveMyImage(outImgPath=None,img=None):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite(outImgPath,img)
        myList=[]
        myList.append(outImgPath)
        
        return myList