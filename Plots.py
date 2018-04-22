import matplotlib.pyplot as plt
class PlotGraph:
    def __init__(self,img):
        self.img=img
    @staticmethod
    def PlotMyImage(img):
        plt.imshow(img)
        plt.show()
    @staticmethod
    def Plot2Images(originalImg,filteredImage):
        plt.subplot(121),plt.imshow(originalImg),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(filteredImage),plt.title('Filtered Image')
        plt.xticks([]), plt.yticks([])
        plt.show()