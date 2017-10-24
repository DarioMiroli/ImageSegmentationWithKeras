from DataGenerator import DataGenerator
import os
dataFolder = "Files/Input/TrainingData/Data"
answersFolder = "Files/Input/TrainingData/Answers"

D1 = DataGenerator(1)
for i in range(10):
    print("On image {}/1000".format(i))
    trainingData, trainingAnswers = D1.generateData(images=1,recNo=100,
            imageWidth=1000,imageHeight=1000)
    fileName = "TrainingRecs{}.png".format(i)
    D1.saveImage(os.path.join(dataFolder,fileName),trainingData[0])
    D1.saveImage(os.path.join(answersFolder,fileName),trainingAnswers[0])
