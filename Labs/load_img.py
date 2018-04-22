import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_salt_and_pepper(gb, prob):
    rnd = np.random.rand(gb.shape[0], gb.shape[1])

    print rnd.shape
    noisy = gb.copy()
    ## for prob = 0.3
    ## >> rnd < 0.3 will be set to 0
    ## >> rnd > 0.7 will be set to 1
    ## >> rnd in between will keep its original values
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy



file_path = "messi5.jpg"
img = cv2.imread(file_path)
# print img[10:15, 10:15,0:3]
cv2.imshow("img",img) #img[50:130,225:260,:]
#cv2.waitKey(0)
#cv2.destroyAllWindows()


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print type(img)
#Another Method:
#img = img[:,:,::-1]
# this means take all elements of the image width, take all elements of the image height,
## take elements of the channels with increment of -1 meaning to reverse it so from BGR it will be RGB

plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#construct Salt and Pepper Noise
salt_noise = add_salt_and_pepper(img, 0.2)
print salt_noise.shape
cv2.imshow("Salt and Pepper Noise",salt_noise)

#construct gaussian noise
gauss_nois = np.zeros(img.shape, dtype=np.uint8)
m = (500,500,500)
s = (500,500,500)
cv2.randn(gauss_nois,m,s)
noised_img = img + gauss_nois
cv2.imshow("gauss_nois",noised_img)

# #construct avg filter
blur = cv2.blur(img, (5, 5))
cv2.imshow("Avg Blurred",blur)

#construct gaussian filter
gblur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("Gaussian Blurred",gblur)

#draw All
plt.figure()
plt.subplot(221), plt.imshow(salt_noise), plt.title('Salt and Pepper Noise')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(noised_img), plt.title('gauss_noise')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(blur), plt.title('Avg Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(gblur), plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
