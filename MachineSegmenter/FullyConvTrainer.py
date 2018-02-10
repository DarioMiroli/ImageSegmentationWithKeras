from FullyConvNet import FullyConvNet
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import sys
import os

#Load or compile model
#Run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = FullyConvNet(1024,1024)
M1.defineModel()
M1.compileModel()
D1 = DataGenerator(1.0)

#Generate Data
folderData = "Files/Input/TrainingData/Data"
folderAnswers = "Files/Input/TrainingData/Answers"
folderValidate = "Files/Input/Images"
outputFolder = "Files/Output"
dataNames = sorted(os.listdir(folderData))
answerNames = sorted(os.listdir(folderAnswers))
validateNames = sorted(os.listdir(folderValidate))
trainingData = []
trainingAnswers = []
for i in range(len(dataNames)):
    #Train model
    trainingData.append(M1.loadImage(os.path.join(folderData,dataNames[i])))
    trainingAnswers.append(M1.loadImage(os.path.join(folderAnswers,answerNames[i])))
    #Threshold training Answers
    #trainingAnswers[trainingAnswers>0] = 1
M1.loadTrainingData(trainingData,trainingAnswers)
D1.displayData(trainingData,delay=0.5)
D1.displayData(trainingAnswers,delay=0.5)


epochs = int(input("Epochs to train?"))
while epochs >0:
    M1.trainModel(batch_size=1,num_epochs=epochs)
    #M1.saveModel("Files/Models/TempModels/MotherMachineTempModel1.h5")
    output = M1.predict(trainingData,threshold=False)
    shouldLoad = input("Ready to see data? (Y)")
    D1.displayData(trainingData,delay=0.1)
    D1.displayData(output,delay=2)
    epochs = int(input("Epochs to train?"))
