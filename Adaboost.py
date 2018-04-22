import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from neupy import algorithms
import os
import cv2
dog_dir = "C:\\Users\\mohamed ismail\\Desktop\\dog\\"
cat_dir = "C:\\Users\\mohamed ismail\\Desktop\\cat\\"
paths_dog_images = [dog_dir+i for i in os.listdir(dog_dir)]
paths_cat_images = [cat_dir+i for i in os.listdir(cat_dir)]
def Hog(img):
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:10, :10], bins[10:,:10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
            for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def LabelsAndFeatures():
    X_dog = []
    X_cat = []
    y_dog = []
    y_cat = []
    for i in range(0, 40):
        if 'dog' in paths_dog_images[i]:
            y_dog.append(1)
            img = cv2.imread(paths_dog_images[i])
            img = cv2.resize(img, (128, 128))
            lst=Hog(img)
            X_dog.append(lst)
        if 'cat' in paths_cat_images[i]:
            y_cat.append(-1)
            img = cv2.imread(paths_cat_images[i])
            img = cv2.resize(img, (128, 128))
            lst=Hog(img)
            X_cat.append(lst)
    X_dog = np.array(X_dog)
    y_dog = np.array(y_dog)
    X_cat = np.array(X_cat)
    y_cat = np.array(y_cat)
    X = np.concatenate((X_dog, X_cat), axis=0)
    y = np.concatenate((y_dog, y_cat))
    return X,y

def PNN(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=1)
    X_train=np.array(X_train,dtype=np.float32)
    X_test=np.array(X_test,dtype=np.float32)
    y_test=np.array(y_test,dtype=np.int32)
    sc=StandardScaler()
    sc.fit(X_train)
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    pnn=algorithms.PNN(std=10,verbose=False)
    pnn.train(X_train,y_train)
    y_predict=pnn.predict(X_test)
    return y_predict,y_test

def Adaboost(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=1)
    X_train=np.array(X_train,dtype=np.float32)
    X_test=np.array(X_test,dtype=np.float32)
    y_test=np.array(y_test,dtype=np.int32)
    sc=StandardScaler()
    sc.fit(X_train)
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),algorithm="SAMME",n_estimators=400)
    bdt.fit(X_train,y_train)
    y_predict=bdt.predict(X_test)
    return y_predict,y_test
def Accuracy(y_predict,y_test):
    count = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_test[i]:
            count += 1
    Accuracy = (count / len(y_predict))*100
    return Accuracy
X,y=LabelsAndFeatures()
y_predict,y_test=PNN(X,y)
acc1=Accuracy(y_predict,y_test)
print("Accuracy= ",acc1,"%")
#HoughTransform("C:\\Users\\mohamed ismail\\Desktop\\messi.jpg")