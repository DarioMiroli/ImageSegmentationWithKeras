from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import sys
import os

#Load or compile model
#Run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = MachineSegmenter()
M1.defineModel(conv_depth_1 = 10, conv_depth_2=10)
M1.compileModel()

#Generate Data
folderData = "Files/Input/TrainingData/Data"
folderAnswers = "Files/Input/TrainingData/Answers"
dataNames = sorted(os.listdir(folderData))
answerNames = sorted(os.listdir(folderAnswers))

for i in range(1):
    #Train model
    trainingData = M1.loadImage(os.path.join(folderData,dataNames[i]))
    trainingAnswers = M1.loadImage(os.path.join(folderAnswers,answerNames[i]))
    trainingData = [trainingData[500*x:500*(1+x),500*y:500*(1+y)]
            for x in range(2) for y in range(2)]
    trainingAnswers = [trainingAnswers[500*x:500*(1+x),500*y:500*(1+y)]
            for x in range(2) for y in range(2)]
    for i in range(1):
        M1.loadTrainingData([trainingData[i]],[trainingAnswers[i]])
        M1.trainModel(batch_size=100,num_epochs=1)
M1.saveModel("Files/Models/TempModels/TempModel1.h5")
output = []
for i in trainingData:
    output.append(M1.predict([i]))
for i in output:
    plt.imshow(i[0],interpolation='none',cmap='gray')
    plt.colorbar()
    plt.show()
