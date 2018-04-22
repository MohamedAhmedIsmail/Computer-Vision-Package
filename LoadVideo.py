import cv2
import numpy as np
videoPath="C:\\Users\mohamed ismail\\Desktop\\Minions_Short_Clip.avi"
videoOutPath="C:\\Users\\mohamed ismail\\Desktop\\OpenCV\\myNewVideo.avi"
videoOutPath2="C:\\Users\\mohamed ismail\\Desktop\\OpenCV\\myNewVideo2.avi"
myImage="C:\\Users\\mohamed ismail\\Desktop\\minions1.PNG"
class Video:
    def ResponseCorrelationVideo(self):
        cap = cv2.VideoCapture(videoPath)
        if cap.isOpened()== False:
            cap.open(videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
        capwriter = cv2.VideoWriter(videoOutPath,fourcc,fps, (int(frame.shape[1]/2), 
                                                              int(frame.shape[0]/2)),isColor=True)
        while(ret):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,dsize=(int(frame.shape[1]/2), int(frame.shape[0]/2)),
                              interpolation=cv2.INTER_CUBIC)
           
            template = cv2.imread(myImage,0)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(gray,template,cv2.TM_CCORR_NORMED)
            res = cv2.resize(res,dsize=(int(frame.shape[1]/2), int(frame.shape[0]/2)),
                              interpolation=cv2.INTER_CUBIC)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            bottom_right = (top_left[0] +w , top_left[1] +h)
            cv2.rectangle(res,top_left, bottom_right,255, 1)
            cv2.putText(res, 'Detected Face', (top_left[0],top_left[1]-10),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255)) 
            cv2.imshow('frame',res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            capwriter.write(cv2.merge([res, res, res]))
            #capwriter.write(gray)
            ret, frame = cap.read()
        cap.release()
        capwriter.release()
        cv2.destroyAllWindows()
        
        
    def MatchTemplateVideo(self):
        cap = cv2.VideoCapture(videoPath)
        if cap.isOpened()== False:
            cap.open(videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
        capwriter = cv2.VideoWriter(videoOutPath2,fourcc,fps, (int(frame.shape[1]/2), 
                                                              int(frame.shape[0]/2)),isColor=True)
        while(ret):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,dsize=(int(frame.shape[1]/2), int(frame.shape[0]/2)),
                              interpolation=cv2.INTER_CUBIC)
            frame = cv2.resize(frame,dsize=(int(frame.shape[1]/2), int(frame.shape[0]/2)),
                              interpolation=cv2.INTER_CUBIC)
            template = cv2.imread(myImage,0)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
            res= cv2.resize(res,dsize=(int(frame.shape[1]), int(frame.shape[0])),
                              interpolation=cv2.INTER_CUBIC)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] +w , top_left[1] +h)
            frame= cv2.rectangle(frame,top_left, bottom_right,255, 1)
            cv2.putText(frame, 'Detected Face', (top_left[0],top_left[1]-10),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255)) 
            
            
            res = (res * 255).astype("uint8")
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
           
            
            #cv2.imshow('frame',frame)
            conc = np.hstack((res, frame))
            cv2.imshow('conc' , conc)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            capwriter.write(cv2.merge([gray, gray, gray]))
            #capwriter.write(gray)
            ret, frame = cap.read()
        cap.release()
        capwriter.release()
        cv2.destroyAllWindows()
        
myVideo=Video()
myVideo.MatchTemplateVideo()