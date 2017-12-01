from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import sys
import os

#Load or compile model
#Run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = MachineSegmenter()
M1.defineModel(conv_depth_1 = 5, conv_depth_2=10,rfSize=21)
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
for i in range(len(dataNames)):
    #Train model
    trainingData = M1.loadImage(os.path.join(folderData,dataNames[i]))
    trainingAnswers = M1.loadImage(os.path.join(folderAnswers,answerNames[i]))
    #Threshold training Answers
    trainingAnswers[trainingAnswers>0] = 1
    M1.loadTrainingData([trainingData],[trainingAnswers])
    M1.trainModel(batch_size=1000,num_epochs=3)

M1.saveModel("Files/Models/TempModels/TempModel1.h5")
output = M1.predict([trainingData],threshold=True)
shouldLoad = raw_input("Ready to see data? (Y)")
D1.displayData([trainingData],delay=3)
D1.displayData(output,delay=5)
