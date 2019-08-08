from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import sys
import os

#Load or compile model
#Run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = MachineSegmenter()
M1.defineModel(conv_depth_1 = 10, conv_depth_2=20,rfSize=21)
M1.compileModel()
D1 = DataGenerator(1.0)

#Generate Data
folderData = "../Files/Input/TrainingData/Data_AgarPads"
folderAnswers = "../Files/Input/TrainingData/Answers_AgarPads"
folderValidate = "../Files/Input/Images"
outputFolder = "Files/Output"
dataNames = sorted(os.listdir(folderData))
answerNames = sorted(os.listdir(folderAnswers))
validateNames = sorted(os.listdir(folderValidate))
trainingImages = []
for i in range(len(dataNames)):
    print("Loaded {0}/{1}".format(i,len(dataNames)-1))
    #Train model
    trainingData = M1.loadImage(os.path.join(folderData,dataNames[i]))
    trainingImages.append(trainingData)
    trainingAnswers = M1.loadImage(os.path.join(folderAnswers,answerNames[i]))
    #D1.displayData([trainingData],delay=0.5)
    #D1.displayData([trainingAnswers],delay=0.5)
    #Threshold training Answers
    trainingAnswers[trainingAnswers>0] = 1
    M1.loadTrainingData([trainingData],[trainingAnswers])
    M1.trainModel(batch_size=1000,num_epochs=1)

M1.saveModel("Models/SlidingWindowModel.h5")
output = M1.predict(trainingImages,threshold=True)
try:
    shouldLoad = raw_input("Ready to see data? (Y)")
except:
    shouldLoad = input("Ready to see data? (Y)")

D1.displayData(trainingImages,delay=2)
D1.displayData(output,delay=2)

M1.plotHistory()
