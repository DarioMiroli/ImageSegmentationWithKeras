from DataGenerator import DataGenerator
import os
dataFolder = "Files/Input/TrainingData/Data"
answersFolder = "Files/Input/TrainingData/Answers"
n = 100
D1 = DataGenerator(1)
for i in range(n):
    print("On image {}/{}".format(i,n))
    trainingData, trainingAnswers = D1.generateData(images=1,recNo=20,
            imageWidth=500,imageHeight=500)
    fileName = "TrainingRecs{}.png".format(i)
    D1.saveImage(os.path.join(dataFolder,fileName),trainingData[0])
    D1.saveImage(os.path.join(answersFolder,fileName),trainingAnswers[0])
