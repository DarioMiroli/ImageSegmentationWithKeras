import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2DTranspose, UpSampling2D, Add, Concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import History
try:
    from scipy.ndimage import imread
except:
    from imageio import imread

try:
    from scipy.misc import imsave
except:
    from imageio import imsave
try:
    from  Tkinter import *
except:
    from tkinter import *
try:
    import Tkinter, Tkconstants, tkFileDialog
except:
    import tkinter
    import tkinter.constants as Tkconstants
    import tkinter.filedialog as tkFileDialog
from keras.utils.vis_utils import plot_model

import sys
import os
import time
import gc

class FullyConvNet:

    def __init__(self,imageWidth,imageHeight,useGPU):
        self.width = imageWidth
        self.height = imageHeight
        self.noClasses = 2
        self.model = None
        self.data = None
        self.answers = None
        self.scores = None
        self.lossHist = []
        self.accHist = []
        self.valLossHist = []
        self.valAccHist = []

        if not useGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def downBlock(self, layer, kernel_size, depth, repeat = 2):
        conv_layer = layer
        for i in range(repeat):
            conv_layer = Convolution2D(depth,(kernel_size,kernel_size),
                    padding='same',activation='relu')(conv_layer)
            norm_layer = BatchNormalization(momentum=0.99)(conv_layer)
        down_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_layer)
        down_pool = Dropout(0.5)(down_pool)
        return norm_layer, down_pool

    def upBlock(self, layer, concat_layer, kernel_size, depth, repeat=2):
        upConvolve_layer = Conv2DTranspose(depth, (2,2),
                strides=2, padding='same')(layer)
        conv_layer = Concatenate()([concat_layer, upConvolve_layer])
        for i in range(repeat):
            conv_layer = Convolution2D(depth,(kernel_size,kernel_size),
                    padding='same',activation='relu')(conv_layer)
        norm_layer = BatchNormalization(momentum=0.99)(conv_layer)
        norm_layer = Dropout(0.5)(norm_layer)
        return norm_layer

    def defineModel(self,kernel_size=3):
            #filterDepthsIn = [1,2,4]
            #filterDepthsIn = [2,4,8]
            filterDepthsIn = [16,32,64,128,256]
            #filterDepthsIn = [10,20,40,80,160]
            filterDepth = [1]
            #filterDepthsIn = [16,32,64]
            #filterDepthsIn = [32,64,128]
            #filterDepthsIn = [64,128,256]
            #filterDepthsIn = [80,160,320]
            #filterDepthsIn = [128,256,512]

            filterDepthsOut = filterDepthsIn[::-1]
            #Input layer
            input_layer = Input(shape=(self.height,self.width,1))
            #Encoding network
            concatenated_layers = []
            l1 = input_layer
            for level in filterDepthsIn:
                conc, l1 = self.downBlock(l1,kernel_size,level)
                concatenated_layers.append(conc)

            for i in range(2):
                l1 = Convolution2D(filterDepthsIn[-1],(kernel_size,kernel_size), padding='same',activation='relu')(l1)

            #Decoder network
            concatenated_layers.reverse()

            for i in range(len(filterDepthsOut)):
                l1 = self.upBlock(l1,concatenated_layers[i],kernel_size, filterDepthsOut[i])
            #Output layer
            output_Layer = Convolution2D(1,(1,1), padding='same',activation='sigmoid')(l1)
            self.model = Model(inputs=input_layer, outputs=output_Layer)

    def compileModel(self):
        try:
            self.model.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
            plot_model(self.model, to_file='./Files/Output/ModelSummaries/ModelSummary.png', show_shapes=True, show_layer_names=True)
        except AttributeError:
            print('Error: Tried to compile model which was undefined')
            sys.exit(0)
        print(self.model.summary())

    def loadTrainingData(self,data,answers,rotateData=True):
        data = self.normaliseData(data)
        #self.displayData(data)
        answers = answers
        #self.displayData(answers,delay=2)
        #Rotate and mirror date if required to pad out training data
        if rotateData:
            if data[0].shape[0] == data[0].shape[1]:
                for i in range(len(data)):
                    for j in range(1,4):
                        data.append(np.rot90(data[i],j))
                        answers.append(np.rot90(answers[i],j))
            for i in range(len(data)):
                data.append(np.fliplr(data[i]))
                data.append(np.flipud(data[i]))
                answers.append(np.fliplr(answers[i]))
                answers.append(np.flipud(answers[i]))
        #Compute scores from answer frames
        scores = self.score(answers)
        #Store data in class varibles
        if self.data == None and self.answers == None:
            self.data = [d for d in data]
            self.answers = [a for a in answers]
            self.scores = [s for s in scores]
        elif self.data != None and self.answers != None:
            self.data = self.data + data
            self.answers = self.answers + answers
            self.scores = self.scores + scores
        else:
            print("Error: Only data or answers have been initialised")
            sys.exit(0)

    def trainModel(self,batch_size=2, num_epochs=1):
        if self.data != None and self.answers != None and self.scores !=None:
            data = np.asarray(self.data,dtype='float16').reshape(len(self.data),
                    self.height, self.width,1)
            scores = np.asarray(self.scores,dtype='float16')
            scores = np.swapaxes(scores,1,3 )
            #Swap axis here so inage is not transposed relative to Training Data
            scores = np.swapaxes(scores,1,2 )
            early_stopping = EarlyStopping(monitor='val_loss', patience=100)
            hist = self.model.fit(data, scores, batch_size=batch_size,
                    epochs=num_epochs, validation_split=0.1, verbose =1,
                    callbacks= [early_stopping])
            print(self.accHist)
            print(hist.history["acc"])
            self.accHist += hist.history["acc"]
            print(self.accHist  )
            self.lossHist += hist.history["loss"]
            self.valAccHist += hist.history["val_acc"]
            self.valLossHist += hist.history["val_loss"]

        else:
            print("Error data or answers not initialised!")
            sys.exit(0)

    def predict(self,images,threshold=False,thresh=0.9):
        predictions = []
        for image in images:
            start = time.time()
            image = self.normaliseData([image])
            image = np.asarray(image).reshape(1,self.height,self.width,1)
            predict = self.model.predict(image)
            i = 0
            print("Prediction completed in {}s".format(time.time()-start))
            predictImage = predict
            if threshold:
                predictImage[predictImage >= thresh] = 1
                predictImage[predictImage < thresh] = 0
            predictions.append(predictImage[0,:,:,0])
        return predictions

    def normaliseData(self,images):
        normalised = []
        for image in images:
            #normal = image/(np.median(image))
            #normal = (image/np.median(image)*2)-1
            normal = 2*((image-np.amin(image))/np.amax(image))-1
            normalised.append(normal)
        return normalised

    def score(self,images):
        classScores = []
        for image in images:
            image[image>0.0] = 1
            image[image<=0.0] = 0
            cellScores = image
            backGroundScores = (image -1)*-1
            classScores.append(np.asarray([cellScores]))
        return classScores

    def saveModel(self,path):
        self.model.save(path)

    def loadModel(self,prompt=True,path=None,title="Select Model"):
        del self.model
        if prompt:
            pathToModel = self.getPathGUI(path,title,File=True)
        else:
            pathToModel = path
        self.model = load_model(pathToModel)
        print(self.model.summary())

    def loadImage(self,path):
        return np.asarray(imread(path),dtype='uint16')

    def loadImagesFromFolder(self,prompt=True, path=None,title="Select folder"):
        if prompt:
            pathToImageFolder = self.getPathGUI(path,title)
        else:
            pathToImageFolder = path
        imageNames = sorted(os.listdir(pathToImageFolder))
        imagePaths = [os.path.join(pathToImageFolder,i) for i in imageNames]
        images = [self.loadImage(i) for i in imagePaths]
        return images, imageNames

    def saveImage(self,path,image):
        imsave(path,image)

    def SaveImagesToFolder(self,images,imageNames,prompt=True,path=None,
            title="Select folder to save to."):
        if prompt:
            pathToSaveFolder = self.getPathGUI(path,title)
        else:
            pathToSaveFolder = path
        for i in range(len(images)):
            self.saveImage(os.path.join(pathToSaveFolder,
                    imageNames[i]),images[i])


    def displayData(self,images,delay=0.5,cmap='gray',):
        plt.ion()
        for image in images:
            plt.clf()
            plt.imshow(image,interpolation='none',cmap=cmap)
            plt.colorbar()
            plt.pause(delay)
        plt.close()

    def getPathGUI(self,path=None,title="",File=False):
        root = Tk()
        if path == None:
            path = "./"
        if not File:
            root.directory = tkFileDialog.askdirectory(initialdir = path,
                    title=title)
            pathToF = root.directory+"/"
        else:
            root.directory = tkFileDialog.askopenfilename(initialdir = path,
                    title=title)
            pathToF = root.directory
        root.withdraw()
        return pathToF

    def plotHistory(self):
        plt.ioff()
        plt.clf()
        plt.close("all")
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        ax.plot(self.lossHist,label="loss",linewidth=3)
        ax.plot(self.valLossHist,label="validation loss",linewidth=3)
        ax.legend(fontsize="xx-large")
        ax.set_xlabel("Epoch",fontsize=25,weight="bold")
        ax.set_ylabel("Loss",fontsize=25,weight="bold")
        ax.tick_params(axis="y", labelsize=20)
        ax.tick_params(axis="x", labelsize=20)
        fig.tight_layout()
        plt.show()
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        ax.plot(self.accHist,label="Accuracy",linewidth=3)
        ax.plot(self.valAccHist,label="validation accuracy",linewidth=3)
        ax.legend(fontsize="xx-large")
        ax.set_xlabel("Epoch",fontsize=25,weight="bold")
        ax.set_ylabel("Accuracy (%)",fontsize=25,weight="bold")
        ax.set_ylim(0.5,1)
        ax.tick_params(axis="y", labelsize=20)
        ax.tick_params(axis="x", labelsize=20)
        fig.tight_layout()
        plt.show()
