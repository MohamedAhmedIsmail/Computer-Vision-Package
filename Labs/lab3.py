import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread("San_Francisco.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("flowers.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("messi5.jpg", cv2.IMREAD_GRAYSCALE)
print img1.shape
print img2.shape
print img3.shape

# img = img2
# h, w = img.shape

# img = cv2.resize(img,  (200, int(200*w/float(h))), interpolation = cv2.INTER_CUBIC)
# print img.shape
#
# remove noise
# img = cv2.GaussianBlur(img,(3,3),0)
#
# # convolute with proper kernels
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
#
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# plt.show()
#
#
print ("====================================================")
#
# img = cv2.imread("San_Francisco.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("flowers.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("messi5.jpg", cv2.IMREAD_GRAYSCALE)
#
# edges = cv2.Canny(img,50,500)
# plt.subplot(121),plt.imshow(img, cmap='gray')
# plt.title("original"), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges, cmap='gray')
# plt.title("edges"), plt.xticks([]), plt.yticks([])
#
# plt.show()