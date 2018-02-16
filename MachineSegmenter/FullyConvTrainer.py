from FullyConvNet import FullyConvNet

#Load or compile model and choose to run on CPU
M1 = FullyConvNet(1024,1024,False)
M1.defineModel()
M1.compileModel()

#Select Folders for training data answers and validation sets
folderData = "Files/Input/TrainingData/Data"
folderAnswers = "Files/Input/TrainingData/Answers"
folderValidate = ("/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk"
    +",share=csce/biology/groups/pilizota/Leonardo_Castorina/"
    + "RDM_TunnelSlide_Slide2_2_Compilled")
outputModelsFolder = "Files/Models/TempModels"

#Get traiing images and answers
trainingData, _ = M1.loadImagesFromFolder(prompt=False,path=folderData)
trainingAnswers, _ = M1.loadImagesFromFolder(prompt=False,path=folderAnswers)

#Load model training data
M1.loadTrainingData(trainingData,trainingAnswers)

#Train for a user defined number of epochs saving model and showing predictions
#each time
epochs = int(raw_input("Epochs to train?"))
while epochs>0:
    M1.trainModel(batch_size=3,num_epochs=epochs)
    M1.saveModel("Files/Models/TempModels/MostRecent_FullyConvNetTempModel1.h5")
    output = M1.predict(trainingData,threshold=False)
    shouldLoad = raw_input("Ready to see data?(Y)")
    M1.displayData(trainingData,delay=0.1)
    M1.displayData(output,delay=2)
    epochs = int(raw_input("Epochs to train?"))

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
