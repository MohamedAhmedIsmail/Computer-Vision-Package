import cv2
import numpy as np
import matplotlib.pyplot as plt
path="C:/Users/mohamed ismail/Desktop/OpenCV/standard_test_images/lena_color_512.tif"
img=cv2.imread(path)
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#img=img[150:400,50:700]
img=cv2.circle(img,(300,280),(100),(255,0,0),4)
"""
cv2.imshow('Lena',img)
k=cv2.waitKey(0)
if k ==27:
    cv2.destroyAllWindows()
elif k== ord('s'):
    cv2.imwrite('grey.png',img)
    cv2.destroyAllWindows()
"""
cv2.imshow("Display",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.imshow(img)
#plt.show()