import cv2
import GrapCutSetting as setting
class MouseEventHandler:
    def __init__(self):
        pass
    def onMouseClick(self,event,x,y,flags,param):
        if event == cv2.EVENT_RBUTTONDOWN:
            setting.rectangle=True
            setting.ix=x
            setting.iy=y
        elif event == cv2.EVENT_MOUSEMOVE:
            if setting.rectangle == True:
                setting.img=setting.img2.copy()
                cv2.rectangle(setting.img,(setting.ix,setting.iy),(x,y),setting.BLUE,2)
                setting.rect=(min(setting.ix,x),min(setting.iy,y),abs(setting.ix-x),abs(setting.iy-y))
                setting.rect_or_mask=0
        elif event == cv2.EVENT_RBUTTONUP:
            print(" Now press the key 'n' a few times until no further change \n")
            setting.rectangle=False
            setting.rect_over=True
            cv2.rectangle(setting.img,(setting.ix,setting.iy),(x,y),setting.BLUE,2)
            setting.rect=(min(setting.ix,x),min(setting.iy,y),abs(setting.ix-x),abs(setting.iy-y))
            setting.rect_or_mask=0
        if event == cv2.EVENT_LBUTTONDOWN:
            if setting.rect_over==False:
                print("Draw the Rectangle First!!")
            else:
                setting.drawing=True
                cv2.circle(setting.img,(x,y),setting.thickness,setting.value['color'],-1)
                cv2.circle(setting.mask,(x,y),setting.thickness,setting.value['val'],-1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if setting.drawing==True:
                cv2.circle(setting.img,(x,y),setting.thickness,setting.value['color'],-1)
                cv2.circle(setting.mask,(x,y),setting.thickness,setting.value['val'],-1)
        elif event == cv2.EVENT_LBUTTONUP:
            if setting.drawing==True:
                setting.drawing=False
                cv2.circle(setting.img,(x,y),setting.thickness,setting.value['color'],-1)
                cv2.circle(setting.mask,(x,y),setting.thickness,setting.value['val'],-1)