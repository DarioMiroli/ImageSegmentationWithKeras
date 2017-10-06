import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy import signal
from scipy.ndimage.filters import gaussian_filter


class DataGenerator:

    def __init__(self,seed):
        np.random.seed(seed)
        self.recs = 50;
        self.image_width = 500
        self.image_height = 500
        self.rec_width= 15
        self.rec_height =5
        self.theta = 360
        self.intensity = 1.0

    def generateSquares(self,images=10):
        squares = []
        for j in range(images):
            image = np.zeros([self.image_width,self.image_height])
            numberOfRecs = int(self.recs*np.random.random())
            for i in range(numberOfRecs):
                temp = np.zeros([self.image_width,self.image_height])
                x_centre = min(max(round(self.image_width*np.random.random()),5),self.image_width-5)
                y_centre = min(max(round(self.image_height*np.random.random()),5),self.image_height-5)
                width = round(max(2*self.rec_width*(np.random.random()+0.5),5))
                height= round(max(2*self.rec_height*(np.random.random()+0.5),5))
                for x in range(self.image_width):
                    for y in range(self.image_height):
                        if x < x_centre + round(width/2.0) and x > x_centre - round(width/2.0):
                            if y < y_centre + round(height/2.0) and y > y_centre - round(height/2.0):
                                temp[x,y] = self.intensity
                theta = self.theta*np.random.random()
                temp = rotate(temp,theta,reshape=False)
                image = np.maximum(temp,image)
                low_values_indices = image < 0.9*self.intensity  # Where values are low
                image[low_values_indices] = 0
                low_values_indices = image > 0.9*self.intensity  # Where values are high
                image[low_values_indices] = self.intensity
            squares.append(image)
        return squares

    def generateData(self,squares):
        blurred = []
        noised = []
        for square in squares:
            blurred.append(gaussian_filter(square, sigma=2))
            noise = np.random.normal(0.5,0.5,self.image_height*self.image_width)
            low_values_indices = noise < 0
            noise[low_values_indices] = 0
            noise = noise.reshape((self.image_width,self.image_height))
            noised.append(blurred[-1]+noise)
        return noised

    def displayData(self,images,delay=0.5,cmap='gray',):
        plt.ion()
        for image in images:
            plt.clf()
            plt.imshow(image,interpolation='none',cmap=cmap)
            plt.colorbar()
            plt.pause(delay)
        plt.close()

    def slice(self,images,size):
        sliced = []
        for image in images:
            width,height = image.shape
            padded = np.pad(image,size,'constant')
            for x in range(size,width+size):
                for y in range(size,height+size):
                    sliced.append(padded[x:x+size,y:y+size])
        return sliced

    def normaliseData(self,images):
        normalised = []
        for image in images:
            normal = image - np.mean(image)
            normalised.append(normal)
        return normalised

    def score(self,images):
        classScores = []
        for image in images:
            xIndex = int(image.shape[0]/2.0)
            yIndex = int(image.shape[1]/2.0)
            if image[xIndex][yIndex] > 0.5:
                classScores.append(np.asarray([1,0]))
            else:
                classScores.append(np.asarray([0,1]))
        return classScores
