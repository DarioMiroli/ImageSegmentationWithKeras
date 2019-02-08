from FullyConvNet import FullyConvNet
import numpy as np
import matplotlib.pyplot as plt

M1 = FullyConvNet(1024,1024,False)

#Darkfield
darks, darkNames = M1.loadImagesFromFolder(prompt=False,path="/home/s1033855/Desktop/DarkFields_31_01_19/Darks_MiddleZoom_Agar_0_1s_Exposure_100_gain")
#Flat fields
flats, flatNames = M1.loadImagesFromFolder(prompt=False,path="/home/s1033855/Desktop/DarkFields_31_01_19/Lights_MiddleZoom_Agar_2s_Exposure_200gain_Flats")
#images
images, imageNames = M1.loadImagesFromFolder(prompt=True)


#Create master dark
masterDark = np.median(np.dstack(darks),axis=2)
M1.displayData([masterDark],delay=2)


#Subtract master dark from flats
flats = [flats[i] - masterDark for i in range(len(flats))]

#Create master flat
masterFlat = np.median(np.dstack(flats),axis=2)
M1.displayData([masterFlat],delay=2)

#Subtract master dark from images
images = [images[i] - masterDark for i in range(len(images))]

#Create corrected images
correctedImages = [(images[i]/masterFlat)*np.median(masterFlat) for i in range(len(images))]

#Save images with corrected prefix
imageNames = ["Corrected_" + imageNames[i] for i in range(len(imageNames))]
M1.SaveImagesToFolder(correctedImages,imageNames)
