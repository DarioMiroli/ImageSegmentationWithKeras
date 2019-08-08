from FullyConvNet import FullyConvNet
import numpy as np
import scipy.ndimage
#Set up and load network using GPU
#M1 = FullyConvNet(1024,1024,True)
#Change this !!!!
M1 = FullyConvNet(1024,1024,True)

#Get Model
#M1.loadModel(prompt=False,path="./Files/Models/SavedModels/Good_Agar_Pad_Model.h5")
M1.loadModel(prompt=False,path="./Files/Models/TempModels/MostRecent_Agar_Pads_Model.h5")

#Select folder to analyse
images, imageNames = M1.loadImagesFromFolder(prompt=False,path="./Files/Input/TrainingData/Data_AgarPads",title= "Select folder to analyse.")

#Pad smaller iamges
for i,image in enumerate(images):
    if image.shape[0] < 1024:
        M1.displayData([image],delay=1)
        xpadBefore = (1024-image.shape[0])/2
        xpadAfter = (1024-image.shape[0])/2
        ypadBefore = (1024-image.shape[1])/2
        ypadAfter = (1024-image.shape[1])/2
        #images[i] = np.pad(image, ((xpadBefore,xpadAfter),(ypadBefore,ypadAfter)), mode="wrap")
        images[i] = scipy.ndimage.zoom(image, 2, order=3)
        M1.displayData([images[i]],delay=1)
        print(images[i].shape)



#images = [x[250:250+512,250:250+512] for x in images]
#images = [x[0:1024,0:1024] for x in images]

#print(images[0].shape)
#M1.displayData(images)

#Predict
predictions = M1.predict(images,threshold=False)

results = []
for i in range(len(images)):
    results.append(images[i])
    results.append(predictions[i])
M1.displayData(results,delay =1)

#Save images
save = raw_input("Save images (y/n)?")
if save == 'y':
    M1.SaveImagesToFolder(images,imageNames,title="Select folder to save to.")
    imageNames = ["prediction"+i for i in imageNames]
    M1.SaveImagesToFolder(predictions,imageNames,title="Select folder to save to.")
