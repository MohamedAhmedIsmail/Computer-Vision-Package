import cv2
import numpy as np
import os
train_dir="C:\\Users\\mohamed ismail\\Desktop\\Train\\"
test_dir="C:\\Users\\mohamed ismail\\Desktop\\Test\\"
paths_train_images = [train_dir+i for i in os.listdir(train_dir)]
paths_test_images = [test_dir+i for i in os.listdir(test_dir)]
def SiftFeaturesExtraction(paths_train_dir,paths_test_dir):
    lst_descriptors_train=[]
    lst_descriptors_test=[]
    for i in paths_train_dir:
        img=cv2.imread(i)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift=cv2.xfeatures2d.SIFT_create()
        _,descriptors=sift.detectAndCompute(gray,None)
        lst_descriptors_train.append(descriptors)
    for j in paths_test_dir:
        img2=cv2.imread(j)
        gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift2=cv2.xfeatures2d.SIFT_create()
        _,descriptors2=sift2.detectAndCompute(gray2,None)
        lst_descriptors_test.append(descriptors2)
    print(lst_descriptors_train)
    return lst_descriptors_train,lst_descriptors_test
def LabelsExtraction(train_dir):
    labels=[]
    for i in os.listdir(train_dir):
        if 'dog' in i:
            labels.append(1)
        else:
            labels.append(0)
    labels=np.asarray(labels)
    return labels
def SVM(labels,training_data,test_data):
    train=np.asarray(training_data)
    test=np.asarray(test_data)
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    for i in range(len(labels)):
        svm.train(train[i][0:len(labels)],cv2.ml.SVM_LINEAR,labels)
    mylst=[]
    for i in range(len(test)):
        sample_data=test_data[i]
        response=svm.predict(sample_data)
        mylst.append(response[1])
    y_predict=np.asarray(mylst)
    total=0
    sums=0
    accuracylist=[]
    for i in range(len(y_predict)):
        total=0
        for j in range(len(labels)):
            if y_predict[i][j]==labels[j] and labels[j]!=0:
                total+=1
        accuracy=(total/y_predict.size)*100
        accuracylist.append(accuracy)
    for i in range(len(accuracylist)):
        sums+=accuracylist[i]
    sums/=len(accuracylist)
    print(sums , "%")
training_data,test_data=SiftFeaturesExtraction(paths_train_images,paths_test_images)
labels=LabelsExtraction(train_dir)
SVM(labels,training_data,test_data)
