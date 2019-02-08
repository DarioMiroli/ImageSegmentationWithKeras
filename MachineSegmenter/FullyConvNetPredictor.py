from FullyConvNet import FullyConvNet

#Set up and load network using GPU
#M1 = FullyConvNet(1024,1024,True)
#Change this !!!!
M1 = FullyConvNet(1024,1024,False)

#Get Model
M1.loadModel()

#Select folder to analyse
images, imageNames = M1.loadImagesFromFolder(title= "Select folder to analyse.")
#images = [x[250:250+512,250:250+512] for x in images]
#images = [x[0:1024,0:1024] for x in images]

#print(images[0].shape)
#M1.displayData(images)

#Predict
predictions = M1.predict(images)

results = []
for i in range(len(images)):
    results.append(images[i])
    results.append(predictions[i])
M1.displayData(results)

#Save images
M1.SaveImagesToFolder(images,imageNames,title="Select folder to save to.")
imageNames = ["prediction"+i for i in imageNames]
M1.SaveImagesToFolder(predictions,imageNames,title="Select folder to save to.")
