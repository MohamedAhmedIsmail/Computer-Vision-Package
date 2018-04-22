import cv2
import numpy as np
img1="C:\\Users\\mohamed ismail\\Desktop\\image_weldo_on_beach.png"
img2="C:\\Users\\mohamed ismail\\Desktop\\img1.png"
class Template:
    def __inti__(self):
        pass
    def TemplateMatching(self):        
        img_rgb = cv2.imread(img1)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(img2,0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.imshow('Detected',img_rgb)
        

myImg=Template()
myImg.TemplateMatching()