import cv2
import numpy as np
import random
    
class FilterImage:
    def __init__(self,img=None):
        self.img=img
    @staticmethod
    def AverageFilter(img=None):
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)
        return dst
    @staticmethod
    def BlurFilter(img=None):
        blur=cv2.blur(img,(13,13))
        return blur
    @staticmethod
    def GaussianFilter(img=None):
        gaussianBlur=cv2.GaussianBlur(img,(13,13),0)
        return gaussianBlur
    @staticmethod
    def MedianFilter(img=None):
        median = cv2.medianBlur(img,13)
        return median
    @staticmethod
    def SaltAndPepper(img=None):
        probability=0.05
        outputImage=np.zeros(img.shape,np.uint8)
        threshold=1-probability
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn=random.random()
                if rdn<probability:
                    outputImage[i][j]=0
                elif rdn>threshold:
                    outputImage[i][j]=255
                else:
                    outputImage[i][j]=img[i][j]
        return outputImage
    @staticmethod
    def GaussianNoise(img=None):
        dest_gauss_noise = np.zeros(img.shape, dtype=np.uint8) 
        m = (0,0,0) 
        s = (50,50,50) 
        cv2.randn(dest_gauss_noise,m,s) 
        noisyImage = img + dest_gauss_noise
        return noisyImage