from FullyConvNet import FullyConvNet

#Set up and load network using GPU
M1 = FullyConvNet(1024,1024,True)

#Get Model
M1.loadModel()

#Select folder to analyse
images, imageNames = M1.loadImagesFromFolder(title= "Select folder to analyse.")
M1.displayData(images)

#Predict
predictions = M1.predict(images)
M1.displayData(predictions)

#Save images
M1.SaveImagesToFolder(predictions,imageNames,title="Select folder to save to.")
