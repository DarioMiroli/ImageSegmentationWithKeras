from FullyConvNet import FullyConvNet
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
from  Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
import sys
import os


#Set up and load network
#   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = FullyConvNet(1024,1024)
root = Tk()
modelPath = tkFileDialog.askopenfilename(initialdir = "./Files/Models/TempModels",
            title = "Select model")
root.withdraw()
M1.loadModel(modelPath)
D1 = DataGenerator(1.0)

#Select folder to analyse
root = Tk()
root.directory = tkFileDialog.askdirectory(initialdir = "./Files/Input/Images",
        title="Select analysis folder")
pathToData = root.directory+"/"
root.withdraw()

#Select output folder
root = Tk()
root.directory = tkFileDialog.askdirectory(initialdir = "./Files/Output",
        title="Select output folder")
outputPath = root.directory+"/"
root.withdraw()


#Get images in folder and show them
imageNames = sorted(os.listdir(pathToData))
imagePaths = [os.path.join(pathToData,i) for i in imageNames]
images = [M1.loadImage(i) for i in imagePaths]
#D1.displayData(images)

#Predict
predictedImages = []
for i,im  in enumerate(images):
    predictedImages.append(M1.predict([im])[0])
    M1.saveImage(outputPath+imageNames[i],predictedImages[-1])
D1.displayData(predictedImages)
