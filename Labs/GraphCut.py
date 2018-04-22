import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Users\\mohamed ismail\\Desktop\\messi.jpg")
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img*mask2[:,:,np.newaxis]
plt.imshow(img2),plt.colorbar(),plt.show()

newmask = cv2.imread("C:\\Users\\mohamed ismail\\Desktop\\Untitled2.png",0)
mask2[newmask == 0] = 0
mask2[newmask == 255] = 1
mask2, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img*mask2[:,:,np.newaxis]
plt.imshow(img2),plt.colorbar(),plt.show()