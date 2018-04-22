import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

class Images:
    
    def readImages(self):
        path="C:/Users/mohamed ismail/Desktop/OpenCV/standard_test_images/lena_color_512.tif"
        img=cv2.imread(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    
    def readImageGray(self):
        path="C:/Users/mohamed ismail/Desktop/OpenCV/standard_test_images/lena_color_512.tif"
        img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
       
class Filters:
    def BlurFilter(self):
        tempimg=Images()
        img=tempimg.readImages()
        blur=cv2.blur(img,(13,13))
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
        plt.show()  
    
    def GaussianFilter(self):
        tempimg=Images()
        img=tempimg.readImages()
        gaussianBlur=cv2.GaussianBlur(img,(13,13),0)
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(gaussianBlur),plt.title('Gaussian Blur')
        plt.show()
        
class Noise:
    def SaltAndPepper(self):
        tempimg=Images()
        img=tempimg.readImages()
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
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(outputImage),plt.title('salt and pepper noise')
        plt.show()   
        
    def GaussianNoise(self):
        tempimg=Images()
        img=tempimg.readImageGray()
        noisySigma=35
        tempImage=np.float64(np.copy(img))
        h=tempImage.shape[0]
        w=tempImage.shape[1]
        noise=np.random.randn(h,w)*noisySigma
        noisyImage=np.zeros(tempImage.shape,np.float64)
        if len(tempImage.shape)==2:
            noisyImage=tempImage+noise
        else:
            noisyImage[:,:,0] = tempImage[:,:,0] + noise
            noisyImage[:,:,1] = tempImage[:,:,1] + noise
            noisyImage[:,:,2] = tempImage[:,:,2] + noise
        cv2.normalize(noisyImage,noisyImage,0,255,cv2.NORM_MINMAX,dtype=-1)
        noisyImage.astype(np.uint8)
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(noisyImage),plt.title('Gaussian Filter Noise')
        plt.show()   
         
        
flter=Filters()        
noise=Noise()
flter.BlurFilter()
flter.GaussianFilter()
noise.SaltAndPepper()
noise.GaussianNoise()        
