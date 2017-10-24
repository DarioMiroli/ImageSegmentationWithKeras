import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as stats
from scipy.misc import imsave
class DataGenerator:

    def __init__(self,seed):
        np.random.seed(int(seed))
        self.recs = 50;
        imageWidth = 500
        imageHeight = 500
        self.rec_width= 15
        self.rec_height =5
        self.theta = 360
        self.intensity = 1.0

    def generateData(self, images=10, imageWidth=500,
            imageHeight=500, recNo=20, recMinWidth=10,
            recMaxWidth=20, recMinHeight=20, recMaxHeight=30,
            blurr=2.0, noiseMagnitude=0.3):

        squares = self.generateSquares(images=images,
                imageWidth=imageWidth,imageHeight=imageHeight,
                recNo=recNo,recMinWidth=recMinWidth,recMaxWidth=
                recMaxWidth, recMinHeight=recMinHeight,
                recMaxHeight=recMaxHeight)

        blurred = self.blurr(squares,blurr=blurr)
        noised = self.noise(blurred,
                noiseMagnitude=noiseMagnitude)

        return noised , squares


    def generateSquares(self,images,imageWidth,imageHeight,
            recNo, recMinWidth,recMaxWidth,recMinHeight,
            recMaxHeight):
        intensity = 1.0
        squares = []
        for j in range(images):
            image = np.zeros([imageWidth,imageHeight])
            numberOfRecs = int(recNo*np.random.random())
            for i in range(numberOfRecs):
                temp = np.zeros([imageWidth,imageHeight])
                x_centre = min(max(round(imageWidth*np.random.random()),5),imageWidth-5)
                y_centre = min(max(round(imageHeight*np.random.random()),5),imageHeight-5)
                width = max(round(recMaxWidth*np.random.random()),recMinWidth)
                height= max(round(recMaxHeight*np.random.random()),recMinHeight)
                for x in range(imageWidth):
                    for y in range(imageHeight):
                        if x < x_centre + round(width/2.0) and x > x_centre - round(width/2.0):
                            if y < y_centre + round(height/2.0) and y > y_centre - round(height/2.0):
                                temp[x,y] = intensity
                theta = 360*np.random.random()
                temp = rotate(temp,theta,reshape=False)
                image = np.maximum(temp,image)
                low_values_indices = image < 0.9*intensity  # Where values are low
                image[low_values_indices] = 0
                low_values_indices = image > 0.9*intensity  # Where values are high
                image[low_values_indices] = intensity
            squares.append(image)
        return squares

    def blurr(self,images,blurr=2):
        blurred = []
        for image in images:
            blurred.append(gaussian_filter(image, sigma=blurr))
        return blurred

    def noise(self,images,noiseMagnitude=2.0):
        noised = []
        stdev = 2.0
        a = 0
        b = 10
        for image in images:
            noise = stats.truncnorm.rvs(a,b,
                    size=np.shape(image)[0]*np.shape(image)[1])
            noise = noise.reshape((np.shape(image)[0],
                    np.shape(image)[1]))
            noised.append(image+noiseMagnitude*noise)
        return noised

    def displayData(self,images,delay=0.5,cmap='gray',):
        plt.ion()
        for image in images:
            plt.clf()
            plt.imshow(image,interpolation='none',cmap=cmap)
            plt.colorbar()
            plt.pause(delay)
        plt.close()

    def saveImage(self,path,image):
        imsave(path,image)
