from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
try:
    from scipy.ndimage import imread
except:
    from imageio import imread

try:
    from scipy.misc import imsave
except:
    from imageio import imsave
import sys
import numpy as np
import time
import gc
from keras.callbacks import EarlyStopping
from keras.callbacks import History
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
from keras.utils.vis_utils import plot_model

class MachineSegmenter:

    def __init__(self):
        self.model = None
        self.data = None
        self.answers = None
        self.scores = None
        self.rfSize = 25
        self.lossHist = []
        self.accHist = []
        self.valLossHist = []
        self.valAccHist = []

    def defineModel(self, num_classes=2, kernel_size=3, pool_size=2,
            conv_depth_1=5, conv_depth_2=3, hidden_size=100, rfSize=21):
        self.rfSize = rfSize
        inp = Input(shape=(rfSize,rfSize,1))
        conv_1 = Convolution2D(conv_depth_1,(kernel_size,kernel_size),
                padding='same',activation='relu')(inp)
        pool_1 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv_1)
        conv_2 = Convolution2D(conv_depth_2,(kernel_size,kernel_size),
                padding='same',activation='relu')(pool_1)
        #pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
        conv_3 = Convolution2D(conv_depth_2,(kernel_size,kernel_size),
                padding='same',activation='relu')(conv_2)
        pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
        flat = Flatten()(pool_3)
        hidden = Dense(hidden_size, activation='relu')(flat)
        #hidden2 = Dense(int(hidden_size/2), activation='relu')(hidden)
        out = Dense(num_classes, activation='softmax')(hidden)
        self.model = Model(inputs=inp, outputs=out)

    def compileModel(self):
        if self.model != None:
            self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
            plot_model(self.model, to_file='Output/ModelSummary.png', show_shapes=True, show_layer_names=True)

        else:
            print('Error: Tried to compile model which was undefined')
            sys.exit(0)

    def loadTrainingData(self,data,answers,asImage=True):
        if asImage:
            data = self.normaliseData(data)
            data = self.slice(data,self.rfSize)
            answers = self.slice(answers,self.rfSize)
            scores = self.score(answers)
        if self.data == None and self.answers == None:
            self.data = [d for d in data]
            self.answers = [a for a in answers]
            if asImage:
                self.scores = [s for s in scores]
        elif self.data != None and self.answers != None:
            self.data = self.data + data
            self.answers = self.answers + answers
            if asImage:
                self.scores = self.scores + scores
        else:
            print("Error: Only data or answers have been initialised")
            sys.exit(0)

    def trainModel(self,batch_size=10, num_epochs=1):
        if self.data != None and self.answers != None and self.scores !=None:
            data = np.asarray(self.data,dtype='uint16').reshape(len(self.data),self.rfSize,self.rfSize,1)
            scores = np.asarray(self.scores,dtype='uint16')
            early_stopping = EarlyStopping(monitor='val_loss', patience=100)
            hist = self.model.fit(data, scores, batch_size=batch_size,
                    epochs=num_epochs, validation_split=0.1, verbose =1,
                    callbacks= [early_stopping])
            self.accHist += hist.history["acc"]
            self.lossHist += hist.history["loss"]
            self.valAccHist += hist.history["val_acc"]
            self.valLossHist += hist.history["val_loss"]
        else:
            print("Error data or answers not initialised!")
            sys.exit(0)
        self.data = None
        self.answers = None

    def slice(self,images,rfSize):
        sliced = []
        if rfSize %2 ==0:
            print("\nRF Size must be odd\n")
            exit()
        nrfSize = rfSize-1
        for image in images:
            width,height = image.shape
            padded = np.pad(image,rfSize,'wrap')
            for x in range(rfSize,width+rfSize):
                for y in range(rfSize,height+rfSize):
                    sliced.append(padded[x-nrfSize//2:x+nrfSize//2+1,
                            y-nrfSize//2:y+nrfSize//2+1])
        return sliced

    def normaliseData(self,images):
        normalised = []
        for image in images:
            normal = image/(np.median(image))
            #normal = 2*((image-np.amin(image))/np.amax(image))-1
            normalised.append(normal)
        return normalised

    def score(self,images):
        classScores = []
        for image in images:
            xIndex = int(image.shape[0]/2.0)
            yIndex = int(image.shape[1]/2.0)
            if image[xIndex][yIndex] > 0:
                classScores.append(np.asarray([1,0]))
            else:
                classScores.append(np.asarray([0,1]))
        return classScores

    def predict(self,images,threshold=False,thresh=0.9):
        predictions = []
        start = time.time()
        for image in images:
            predictImage = np.zeros(np.shape(image),dtype='float16')
            image = self.normaliseData([image])
            image = self.slice(image,self.rfSize)
            image = np.asarray(image).reshape(len(image),self.rfSize,
                    self.rfSize,1)
            print("Starting Prediction", time.time()-start)
            predict = self.model.predict(image)
            i = 0
            print("Prediction complete", time.time()-start)
            for x in range(predictImage.shape[0]):
                for y in range(predictImage.shape[1]):
                    predictImage[x][y] = predict[i][0]
                    i+=1
            if threshold:
                predictImage[predictImage>=0.9*thresh] = 1
                predictImage[predictImage<0.9*thresh] = 0
            predictions.append(predictImage)
        return predictions

    def saveModel(self,path):
        self.model.save(path)

    def loadModel(self,path):
        del self.model
        self.model = load_model(path)

    def loadImage(self,path):
        return np.asarray(imread(path),dtype='uint16')

    def saveImage(self,path,image):
        imsave(path,image)

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
