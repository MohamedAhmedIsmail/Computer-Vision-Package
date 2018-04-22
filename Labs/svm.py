import cv2
import numpy as np

trainingData = np.array([[501, 10], [255, 10], [501, 255], [10, 501]], np.float32)
labels = np.array([1, -1, -1, -1])

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setDegree(0.0)
# svm.setGamma(0.0)
# svm.setCoef0(0.0)
# svm.setC(0)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)
# svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))

svm.train(trainingData, cv2.ml.SVM_LINEAR, labels)

sample_data = np.array([[10,10]], np.float32)
response = svm.predict(sample_data)

height = 512
width = 512
image = np.zeros((height, width, 3));
green, blue = (0, 255, 0), (255, 0, 0)

# Show the decision regions given by the SVM
for i in range(image.shape[1]):
    for j in range(image.shape[0]):
        sample_data = np.array([[j, i]], np.float32)
        response = svm.predict(sample_data)[1].ravel();

        if response == 1:
            image[i, j] = green;
        elif response == -1:
            image[i, j]  = blue;


# Show the training data

thickness = -1;
lineType = 8;
cv2.circle(image, (501, 10), 5, (0, 0, 0), thickness, lineType);
cv2.circle(image, (255, 10), 5, (255, 255, 255), thickness, lineType);
cv2.circle(image, (501, 255), 5, (255, 255, 255), thickness, lineType);
cv2.circle(image, (10, 501), 5, (255, 255, 255), thickness, lineType);

cv2.imwrite("result.png", image) # save the image

cv2.imshow("SVM Simple Example", image) # show it to the user
cv2.waitKey(0)