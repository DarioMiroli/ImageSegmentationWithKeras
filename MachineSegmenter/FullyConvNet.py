import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2DTranspose, UpSampling2D, Add, Concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from scipy.ndimage import imread
from scipy.misc import imsave
from  Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
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

    def downBlock(self, layer, kernel_size, depth, repeat = 3):
        l1 = layer
        for i in range(repeat):
            conv_layer = Convolution2D(depth,(kernel_size,kernel_size),
                    padding='same',activation='relu')(l1)
            norm_layer = BatchNormalization(momentum=0.99)(conv_layer)
        down_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_layer)
        return norm_layer, down_pool

    def upBlock(self, layer, concat_layer, kernel_size, depth, repeat=3):
        upConvolve_layer = Conv2DTranspose(depth, (kernel_size,kernel_size),
                strides=2, padding='same')(layer)
        concatenated_layer = Concatenate()([concat_layer, upConvolve_layer])
        norm_layer = BatchNormalization(momentum=0.99)(concatenated_layer)
        for i in range(repeat):
            conv_layer = Convolution2D(depth,(kernel_size,kernel_size),
                    padding='same',activation='relu')(norm_layer)
            norm_layer = BatchNormalization(momentum=0.99)(conv_layer)
        return norm_layer

    def defineModel(self,kernel_size=3):
            #filterDepthsIn = [1,2,4]
            #filterDepthsIn = [2,4,8]
            #filterDepthsIn = [4,8,16]
            filterDepthsIn = [8,16,32]
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
            #Decoder network
            concatenated_layers.reverse()

            for i in range(len(filterDepthsOut)):
                l1 = self.upBlock(l1,concatenated_layers[i],kernel_size, filterDepthsOut[i])
            #Output layer
            output_Layer = Convolution2D(1,(3,3), padding='same',activation='sigmoid')(l1)
            self.model = Model(inputs=input_layer, outputs=output_Layer)

    def compileModel(self):
        try:
            self.model.compile(loss='mean_squared_error',
                    optimizer='adam', metrics=['accuracy'])
        except AttributeError:
            print('Error: Tried to compile model which was undefined')
            sys.exit(0)
        print(self.model.summary())

    def loadTrainingData(self,data,answers,rotateData=True):
        data = self.normaliseData(data)
        answers = answers
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
            normal = (image/np.median(image)*2)-1
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
        plt.plot(self.lossHist,label="loss")
        plt.plot(self.valLossHist,label="val loss")
        plt.legend()
        plt.show()
        plt.plot(self.accHist,label="Acc")
        plt.plot(self.valAccHist,label="val acc")
        plt.legend()
        plt.show()
