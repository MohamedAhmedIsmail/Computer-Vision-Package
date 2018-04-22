import numpy as np
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread("C:\\Users\\mohamed ismail\\Desktop\\query.jpg",0)          # queryImage
 # trainImage
myListImage=["C:\\Users\\mohamed ismail\\Desktop\\1.jpg","C:\\Users\\mohamed ismail\\Desktop\\2.jpg","C:\\Users\\mohamed ismail\\Desktop\\3.jpg",
             "C:\\Users\\mohamed ismail\\Desktop\\4.jpg","C:\\Users\\mohamed ismail\\Desktop\\5.jpg","C:\\Users\\mohamed ismail\\Desktop\\6.jpg",
             "C:\\Users\\mohamed ismail\\Desktop\\7.jpg","C:\\Users\\mohamed ismail\\Desktop\\8.jpg","C:\\Users\\mohamed ismail\\Desktop\\query.jpg"]
print(myListImage)
mySumList=[]
myImgList=[]
# Initiate SIFT detector
for i in range(len(myListImage)):
    img2 = cv2.imread(myListImage[i],0)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
        
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    mysum=0.0
    for m in matches:
        mysum+=m.distance
        
    mySumList.append(mysum)
    img3 = cv2.drawMatches(img1,kp1,
                           img2,kp2,
                           matches[:10],
                           flags=2, outImg=None)
    myImgList.append(img3)
myMinimumIndex=np.argmin(mySumList)
# Draw first 10 matches.
for i in range(len(myImgList)):
    plt.imshow(myImgList[i]),plt.show()
    print("distance= ",mySumList[i])
    
maxDistance=np.max(mySumList)
print(myImgList[maxDistance])
#plt.imshow(myImgList[myMinimumIndex]),plt.show()
