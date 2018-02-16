from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
import sys
import os

useGPU = raw_input("Use GPU device? (y/n)")
if not useGPU == "y":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Deal with input output folders
inputFolder = "./Files/Input/Images"
outputFolder  = "./Files/Output"
fileNames = os.listdir(inputFolder)

#Load or compile model
shouldLoad = raw_input("Load model? (y/n)")
M1 = MachineSegmenter()
if shouldLoad == 'y':
    M1.loadModel("MyModel.h5")
else:
    M1.defineModel(conv_depth_1 = 10, conv_depth_2=1)
    M1.compileModel()

#Train model or not
shouldTrain = raw_input("Train model? (y/n)")
D1 = DataGenerator(1.0)
if shouldTrain == 'y':
    #Generate Data
    trainingData, trainingAnswers = D1.generateData(images=2,recNo=20)
    D1.displayData(trainingData,1)
    D1.displayData(trainingAnswers,1)
    #Train model
    M1.loadTrainingData(trainingData,trainingAnswers)
    M1.trainModel(batch_size=1000,num_epochs=3)

#Run model on validation generated data
shouldValidate = raw_input("\nValidate model on a fresh set of generated data?")
if shouldValidate == 'y':
    validationData , _ = D1.generateData(images=1,recNo = 20)
    validationPredicts = M1.predict(validationData)
    D1.displayData(validationData, delay=3)
    D1.displayData(validationPredicts, delay=3)

#Run model on images
shouldTest = raw_input("Test model on images in input folder? (y/n)")
if shouldTest == 'y':
    imagePredictions = []
    images = []
    for f in fileNames:
        image = M1.loadImage(os.path.join(inputFolder,f))
        images.append(image)
        imagePredictions = imagePredictions + M1.predict([image])

    D1.displayData(images)
    D1.displayData(imagePredictions,5)

    #save images
    shouldSaveImages = raw_input("Save images to output folder? (y/n)")
    if shouldSaveImages == 'y':
        for i, image in enumerate(imagePredictions):
            M1.saveImage(os.path.join(outputFolder,fileNames[i]),image)

#Save model
shouldSave = raw_input("Save model? (y/n)")
if shouldSave == 'y':
    M1.saveModel("MyModel.h5")
