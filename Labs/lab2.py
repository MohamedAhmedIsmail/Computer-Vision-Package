import numpy as np
import kernel_plotter as kp
import cv2
import argparse


a = np.array([1, 2, 3, 4, 5])
print a.sum()
print np.sum(a)

a = np.ones((3, 4, 2))
print a.sum()
print a.sum(axis=0)
print a.sum(axis=1)
print a.sum(axis=2)

a = np.array([[1, 1], [2, 2]])
b = np.array([[2, 2], [3, 3]])
print a.dot(b)

print "============================================="

def rescale_intensity(img):
    imin=0
    imax=255
    omin=0
    omax=1
    img = np.clip(img, imin, imax)
    image = (img - imin) / float(imax - imin)
    return image * (omax - omin) + omin

def correlation(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) / 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=0)
    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual correlation by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output)
    output = (output*255).astype("uint8")
    return output

# # construct average blurring kernels used to smooth an image
shiftR = np.zeros((3, 3), dtype="float"); shiftR[1,0] = 1
shiftT = np.zeros((3, 3), dtype="float"); shiftT[2,1] = 1
smallBlur = np.ones((3, 3), dtype="float") * (1.0 / (3*3))
largeBlur = np.ones((11, 11), dtype="float") * (1.0 / (11 * 11))
largerBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
size_kernel = 3
std_dev = 1
gaussianBlurX= cv2.getGaussianKernel(size_kernel,std_dev , cv2.CV_32F )
gaussianBlurY= cv2.getGaussianKernel(size_kernel,std_dev , cv2.CV_32F )
gaussianBlur = np.dot(gaussianBlurX,gaussianBlurY.T)

kernelBank = [
	("shift_right", shiftR),
	("shift_top", shiftT),
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("larger_blur", largerBlur),
    ("gaussian_blur", gaussianBlur)
    ]

img = cv2.imread("messi5.jpg",0)

for kernelName,kernel in kernelBank:
    corrOutput = correlation(img, kernel)
    # cv2.imshow("original", img)
    cv2.imshow("{} - convole".format(kernelName), corrOutput)
    kp.plot(kernel, kernelName)

cv2.waitKey(0)
cv2.destroyAllWindows()

print "-----------------------------------"

Sharpining by Averaging
avg_img = correlation(img, gaussianBlur)
details = img - avg_img
cv2.imshow("details", details)
img_sharpened = img + details
cv2.imshow("img sharpened", img_sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
#
print "-----------------------------------"

#Sharpining by Correlation
sharpen = np.array([
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]], dtype="int")

img_sharpened2 = correlation(img, sharpen)

cv2.imshow("img sharpened2", img_sharpened2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print "--------------------------------------------------"

# cap = cv2.VideoCapture(0) # for camera
cap = cv2.VideoCapture("Minions Short Clip.mp4")
if cap.isOpened()== False:
    cap.open("Minions Short Clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
ret, frame = cap.read()
aspect_ratio = frame.shape[0]/float(frame.shape[1])
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #oR cv2.VideoWriter_fourcc(*'MJPG)
capwriter = cv2.VideoWriter('Minion Gray.avi',fourcc,
                            fps, (frame.shape[1]/2, frame.shape[0]/2) ,
                            isColor = False) #isColor works only on windows

while(ret):
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,
                      dsize=(frame.shape[1]/2, frame.shape[0]/2),
                      interpolation=cv2.INTER_CUBIC)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # capwriter.write(cv2.merge([gray, gray, gray])) # in case isColor is set to true
    capwriter.write(gray)
    # Capture frame-by-frame
    ret, frame = cap.read()

# When everything done, release the capture
cap.release()
capwriter.release()
cv2.destroyAllWindows()