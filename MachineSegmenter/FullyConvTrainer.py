from FullyConvNet import FullyConvNet
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import sys
import os

#Load or compile model
#Run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
M1 = FullyConvNet(1024,1024)
M1.defineModel()
M1.compileModel()
D1 = DataGenerator(1.0)

#Generate Data
folderData = "Files/Input/TrainingData/Data"
folderAnswers = "Files/Input/TrainingData/Answers"
folderValidate = ("/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk,"
    +"share=csce/biology/groups/pilizota/Leonardo_Castorina/"
    + "RDM_TunnelSlide_Slide2_2_Compilled")

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


epochs = int(raw_input("Epochs to train?"))
while epochs >0:
    M1.trainModel(batch_size=3,num_epochs=epochs)
    M1.saveModel("Files/Models/TempModels/FullyConvNetTempModel1.h5")
    output = M1.predict(trainingData,threshold=False)
    shouldLoad = raw_input("Ready to see data?(Y)")
    D1.displayData(trainingData,delay=0.1)
    D1.displayData(output,delay=2)
    epochs = int(raw_input("Epochs to train?"))

validateData = []
for i in range(len(validateNames)):
    #Train model
    validateData.append(M1.loadImage(os.path.join(folderValidate,validateNames[i])))
    #Threshold training Answers
    #trainingAnswers[trainingAnswers>0] = 1
output = M1.predict(validateData,threshold=True,thresh=0.5)

validationDisplayData = []
j=1
for i in range(len(validateData)*2-1):
    print(i,j)
    if i % 2 == 0:
        validationDisplayData.append(output[j-1])
        j += 1
    else:
        validationDisplayData.append(validateData[j-1])

D1.displayData(validationDisplayData,delay=2)
