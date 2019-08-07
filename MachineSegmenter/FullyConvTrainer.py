from FullyConvNet import FullyConvNet
import numpy as np
#Load or compile model and choose to run on CPU
M1 = FullyConvNet(1024,1024,True)
M1.defineModel()
M1.compileModel()

#Select Folders for training data answers and validation sets
if False:
    folderData = "Files/Input/TrainingData/Data_Flourescent"
    folderAnswers = "Files/Input/TrainingData/Answers_Flourescent"
    folderValidate = ("/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk"
        +",share=csce/biology/groups/pilizota/Leonardo_Castorina/"
        + "RDM_TunnelSlide_Slide2_2_Compilled")
    outputModelsFolder = "Files/Models/TempModels"

else:
    folderData = "Files/Input/TrainingData/Data_AgarPads"
    folderAnswers = "Files/Input/TrainingData/Answers_AgarPads"
    #folderValidate = ("/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk"
    #    +",share=csce/biology/groups/pilizota/Leonardo_Castorina/"
    #    + "RDM_TunnelSlide_Slide2_2_Compilled")
    folderValidate = "Files/Input/TrainingData/Data_Flourescent"#"/home/s1033855/Desktop/DarkFields_31_01_19/CorrectedHighDensityCells"
    outputModelsFolder = "Files/Models/TempModels"

#Get traiing images and answers
trainingData, _ = M1.loadImagesFromFolder(prompt=False,path=folderData)
#trainingData = [x[250:250+512,250:250+512] for x in trainingData]
trainingAnswers, _ = M1.loadImagesFromFolder(prompt=False,path=folderAnswers)
trainingAnswers = [np.asarray(i > 0,dtype="bool")*1 for i in trainingAnswers]
#trainingAnswers = [x[250:250+512,250:250+512] for x in trainingAnswers]
#Load model training data
for i in range(len(trainingData)):
    #M1.displayData([trainingData[i]])
    #M1.displayData([trainingAnswers[i]])
    pass
#M1.displayData(trainingAnswers)

M1.loadTrainingData(trainingData,trainingAnswers,True)
#Train for a user defined number of epochs saving model and showing predictions
#each time
try:
    epochs = int(raw_input("Epochs to train?"))
except:
    epochs =  int(input("Epochs to train?"))
while epochs>0:
    M1.trainModel(batch_size=3,num_epochs=epochs)
    M1.saveModel("Files/Models/TempModels/MostRecent_Agar_Pads_Model.h5")
    output = M1.predict(trainingData,threshold=False)
    try:
        shouldLoad =  (raw_input("Ready to see data? (y) "))
    except:
        shouldLoad =  (input("Ready to see data? (y) "))
    M1.displayData(trainingData,delay=0.1)
    M1.displayData(output,delay=2)
    M1.plotHistory()
    try:
        epochs = int(raw_input("Epochs to train?"))
    except:
        epochs =  int(input("Epochs to train?"))

#Test on validation data sets
validateData, _ = M1.loadImagesFromFolder(prompt=False,path=folderValidate)
output = M1.predict(validateData,threshold=True,thresh=0.5)

#Show inputand outputs interleaved
validationDisplayData = []
j=1
for i in range(len(validateData)*2-1):
    if i % 2 == 0:
        validationDisplayData.append(output[j-1])
        j += 1
    else:
        validationDisplayData.append(validateData[j-1])

M1.displayData(validationDisplayData,delay=2)
