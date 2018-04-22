import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
class OrientationGradient:
    def __init__(self):
        pass
    def reshape2DArray(self,arr, nrows, ncols):
        h, w = arr.shape
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    def Task(self,imgPath,D):
        img=cv2.imread(imgPath,0)
        h,w=img.shape
        print(h)
        print()
        print(w)
        #img = cv2.resize(img,(200,int(200*h/float(w))),interpolation=cv2.INTER_CUBIC)
        gaussianBlur=cv2.GaussianBlur(img,(5,5),0)
        sobelx = cv2.Sobel(gaussianBlur,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(gaussianBlur,cv2.CV_64F,0,1,ksize=5)
        gradient2D=[]
        
        for i in range(len(sobelx)):
            gradient=[]
            for j in range(len(sobelx[i])):
                directionGradient=math.atan2(sobely[i][j],sobelx[i][j])
                Angle=(directionGradient*180)/math.pi
                gradient.append(Angle)
            gradient2D.append(gradient)

        npgradient=np.array(gradient2D)
        print(npgradient)
        print()
        D1=h/D
        D2=w/D
        medianArr=self.reshape2DArray(npgradient,D1,D2)
        resnp=np.array(medianArr)
        print(resnp)
        print(len(resnp))
        print()
        res=[]
        for i in range(len(resnp)):
            mylist=[]
            mylist.append(np.median(medianArr[i]))
            res.append(mylist)
        
        newimg=np.array(res)
        print(newimg)
        plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
myObj=OrientationGradient()
myObj.Task("C:\\Users\\mohamed ismail\\Desktop\\messi.jpg",2)