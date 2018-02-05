from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import sys
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = MachineSegmenter()
M1.loadModel("./Files/Models/TempModels/MotherMachineTempModel1.h5")
M1.compileModel()
D1 = DataGenerator(1.0)
folderValidate = "Files/Input/Images"
validateNames = sorted(os.listdir(folderValidate))

outputFolder = "Files/Output"
imagePredictions = []
images = []
for f in validateNames[2:4]:
    image = M1.loadImage(os.path.join(folderValidate,f))
    images.append(image)
    imagePredictions = imagePredictions + M1.predict([image],threshold=True,thresh=0.5)
D1.displayData(images,3)
D1.displayData(imagePredictions,5)

for i,im in enumerate(imagePredictions):
    M1.saveImage(os.path.join(outputFolder,validateNames[i]),im)
