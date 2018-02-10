from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2DTranspose, UpSampling2D, Add, Concatenate, concatenate
from keras.callbacks import EarlyStopping
from scipy.ndimage import imread
from scipy.misc import imsave
import sys
import numpy as np
import time
import gc

#Delete these
import matplotlib.pyplot as plt

class FullyConvNet:

    def __init__(self,imageWidth,imageHeight):
        self.width = imageWidth
        self.height = imageHeight
        self.noClasses = 2
        self.model = None
        self.data = None
        self.answers = None
        self.scores = None

    def downBlock(self, depth, kernel_size, lay, mpool = True):
        b = lay
        for i in range(3):
            c = Convolution2D(depth,(kernel_size,kernel_size), padding='same',activation='relu')(b)
            b = BatchNormalization(momentum=0.99)(c)

        m = MaxPooling2D(pool_size=(2, 2), strides=2)(b)
        return b, m

    def upBlock(self, depth, kernel_size, lay, conc_layer):
        u = Conv2DTranspose(depth, (kernel_size,kernel_size), strides=2, padding='same')(lay)
        con = Concatenate()([conc_layer, u])
        b = BatchNormalization(momentum=0.99)(con)
        for i in range(3):
            c = Convolution2D(depth,(kernel_size,kernel_size), padding='same',activation='relu')(b)
            b = BatchNormalization(momentum=0.99)(c)
        return b

    def defineModel(self,kernel_size=3):
            filterDepths = [16,32,64]
            filterDepthsOut = filterDepths[::-1]

            print("Filters:")
            print (filterDepths)
            print (filterDepthsOut)
            conc_layers = []

            self.model = Sequential()

            inputLayer = Input(shape = (self.width,self.height,1))
            inLay = inputLayer


            for lev in filterDepths:
                conc, inLay = self.downBlock(lev, kernel_size, inLay)
                conc_layers.append(conc)

            conc_layers.reverse()

            for i in range(len(filterDepthsOut)):
                inLay = self.upBlock(filterDepthsOut[i], kernel_size, inLay, conc_layers[i])

            outputLayer = Convolution2D(1,(1,1), padding='same',activation='sigmoid')(inLay)

            self.model = Model(inputs=inputLayer, outputs=outputLayer)

            # #Encoder network
            # # 3 convolve and down 1
            # conv_1 = Convolution2D(filterDepths[0],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(inp)
            # norm_1 = BatchNormalization(momentum=0.99)(conv_1)
            #
            # conv_2 = Convolution2D(filterDepths[0],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_1)
            # norm_2 = BatchNormalization(momentum=0.99)(conv_2)
            #
            # conv_3 = Convolution2D(filterDepths[0],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_2)
            # norm_3 = BatchNormalization(momentum=0.99)(conv_3)
            #
            # pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_3)
            #
            # # 3 convolve and down 2
            # conv_4 = Convolution2D(filterDepths[1],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(pool_1)
            # norm_4 = BatchNormalization(momentum=0.99)(conv_4)
            #
            # conv_5 = Convolution2D(filterDepths[1],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_4)
            # norm_5 = BatchNormalization(momentum=0.99)(conv_5)
            #
            # conv_6 = Convolution2D(filterDepths[1],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_5)
            # norm_6 = BatchNormalization(momentum=0.99)(conv_6)
            #
            # pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_6)
            #
            # #3 convolve and down 3
            # conv_7 = Convolution2D(filterDepths[2],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(pool_2)
            # norm_7 = BatchNormalization(momentum=0.99)(conv_7)
            #
            # conv_8 = Convolution2D(filterDepths[2],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_7)
            # norm_8 = BatchNormalization(momentum=0.99)(conv_8)
            #
            # conv_9 = Convolution2D(filterDepths[2],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_8)
            # norm_9 = BatchNormalization(momentum=0.99)(conv_9)
            #
            # pool_3 = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_9)
            #
            # #3 convolve and down 4
            # conv_10 = Convolution2D(filterDepths[3],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(pool_3)
            # norm_10 = BatchNormalization(momentum=0.99)(conv_10)
            #
            # conv_11 = Convolution2D(filterDepths[3],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_10)
            # norm_11 = BatchNormalization(momentum=0.99)(conv_11)
            #
            # conv_12 = Convolution2D(filterDepths[3],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_11)
            # norm_12 = BatchNormalization(momentum=0.99)(conv_12)
            #
            # pool_4 = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_12)
            #
            # #3 convolve and down 5
            # conv_13 = Convolution2D(filterDepths[4],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(pool_4)
            # norm_13 = BatchNormalization(momentum=0.99)(conv_13)
            #
            # conv_14 = Convolution2D(filterDepths[4],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_13)
            # norm_14 = BatchNormalization(momentum=0.99)(conv_14)
            #
            # conv_15 = Convolution2D(filterDepths[4],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_14)
            # norm_15 = BatchNormalization(momentum=0.99)(conv_15)
            #
            # pool_5 = MaxPooling2D(pool_size=(2, 2), strides=2)(norm_15)
            #
            # #3 convolve Across
            # conv_16 = Convolution2D(filterDepths[5],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(pool_5)
            # norm_16 = BatchNormalization(momentum=0.99)(conv_16)
            #
            # conv_17 = Convolution2D(filterDepths[5],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_16)
            # norm_17 = BatchNormalization(momentum=0.99)(conv_17)
            #
            # conv_18 = Convolution2D(filterDepths[5],(kernel_size,kernel_size),
            #         padding='same',activation='relu')(norm_17)
            # norm_18 = BatchNormalization(momentum=0.99)(conv_18)
            #
            #
            #
            #
            # #Decoder network
            #
            # #Up pool concatenate and covolve twice. 1
            # upPool_1 = UpSampling2D()(norm_18)
            # merge_1 = Concatenate()([upPool_1, norm_15])
            #
            # deConv_1 = Conv2DTranspose(filterDepths[4],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(merge_1)
            # norm_19 = BatchNormalization(momentum=0.99)(deConv_1)
            #
            # deConv_2 = Conv2DTranspose(filterDepths[4],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_19)
            # norm_20 = BatchNormalization(momentum=0.99)(deConv_2)
            #
            # deConv_3 = Conv2DTranspose(filterDepths[4],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_20)
            # norm_21 = BatchNormalization(momentum=0.99)(deConv_3)
            #
            # #Up pool concatenate and covolve twice. 2
            # upPool_2 = UpSampling2D()(norm_21)
            # merge_2 = Concatenate()([upPool_2, norm_12])
            #
            # deConv_4 = Conv2DTranspose(filterDepths[3],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(merge_2)
            # norm_22 = BatchNormalization(momentum=0.99)(deConv_4)
            #
            # deConv_5 = Conv2DTranspose(filterDepths[3],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_22)
            # norm_23 = BatchNormalization(momentum=0.99)(deConv_5)
            #
            # deConv_6 = Conv2DTranspose(filterDepths[3],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_23)
            # norm_24 = BatchNormalization(momentum=0.99)(deConv_6)
            #
            # #Up pool concatenate and covolve twice. 3
            # upPool_3 = UpSampling2D()(norm_24)
            # merge_3 = Concatenate()([upPool_3, norm_9])
            #
            # deConv_7 = Conv2DTranspose(filterDepths[2],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(merge_3)
            # norm_25 = BatchNormalization(momentum=0.99)(deConv_7)
            #
            # deConv_8 = Conv2DTranspose(filterDepths[2],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_25)
            # norm_26 = BatchNormalization(momentum=0.99)(deConv_8)
            #
            # deConv_9 = Conv2DTranspose(filterDepths[2],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_26)
            # norm_27 = BatchNormalization(momentum=0.99)(deConv_9)
            #
            #
            # #Up pool concatenate and covolve twice. 4
            # upPool_4 = UpSampling2D()(norm_27)
            # merge_4 = Concatenate()([upPool_4, norm_6])
            #
            # deConv_10 = Conv2DTranspose(filterDepths[1],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(merge_4)
            # norm_28 = BatchNormalization(momentum=0.99)(deConv_10)
            #
            # deConv_11 = Conv2DTranspose(filterDepths[1],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_28)
            # norm_29 = BatchNormalization(momentum=0.99)(deConv_11)
            #
            # deConv_12 = Conv2DTranspose(filterDepths[1],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_29)
            # norm_30 = BatchNormalization(momentum=0.99)(deConv_12)
            #
            # #Up pool concatenate and covolve twice. 5
            # upPool_5 = UpSampling2D()(norm_30)
            # merge_5 = Concatenate()([upPool_5, norm_3])
            #
            # deConv_13 = Conv2DTranspose(filterDepths[0],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(merge_5)
            # norm_31 = BatchNormalization(momentum=0.99)(deConv_13)
            #
            # deConv_14 = Conv2DTranspose(filterDepths[0],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_31)
            # norm_32 = BatchNormalization(momentum=0.99)(deConv_14)
            #
            # deConv_15 = Conv2DTranspose(filterDepths[0],(kernel_size,kernel_size),
            #         padding="same",activation = 'relu')(norm_32)
            # norm_33 = BatchNormalization(momentum=0.99)(deConv_15)
            #
            #
            # #Output
            # out = Convolution2D(1,(kernel_size,kernel_size),activation='sigmoid'
            #     ,padding='same')(norm_33)
            #

    def compileModel(self):
        if self.model != None:
            #self.model.compile(loss='binary_crossentropy',
            #        optimizer='adam', metrics=['accuracy'])
            self.model.compile(loss='mean_squared_error',
                    optimizer='adam', metrics=['accuracy'])
            print(self.model.summary())
        else:
            print('Error: Tried to compile model which was undefined')
            sys.exit(0)

    def loadTrainingData(self,data,answers):
        data = self.normaliseData(data)
        answers = answers
        scores = self.score(answers)
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
        self.fullData = None
        self.fullScores = None

    def trainModel(self,batch_size=2, num_epochs=1):
        if self.data != None and self.answers != None and self.scores !=None:
            data = np.asarray(self.data,dtype='uint16').reshape(len(self.data),self.width,
                    self.height,1)
            scores = np.asarray(self.scores,dtype='uint16')
            scores = np.swapaxes(scores,1,3 )
            #hmmmm not sure about this next line
            scores = np.swapaxes(scores,1,2 )

            if self.fullData == None or self.fullScores == None:
                fullData, fullScores = self.increaseData(data, scores)

            #early_stopping = EarlyStopping(monitor='loss', patience=3)
            self.model.fit(fullData, fullScores, batch_size=batch_size,
                    epochs=num_epochs, validation_split=0.1, verbose =1)
        else:
            print("Error data or answers not initialised!")
            sys.exit(0)
        #self.data = None
        #self.answers = None

    def increaseData(self, data, scores):
        fullData = np.zeros((data.shape[0]*12, data.shape[1], data.shape[2], data.shape[3]))
        fullScores = np.zeros((scores.shape[0]*12, scores.shape[1], scores.shape[2], scores.shape[3]))
        for i in range(data.shape[0]):
            im = data[i]
            ans = scores[i]
            for j in range(4):
                index = (i*12)+(j*3)
                #Rotate data and take mirror of all rotations
                dat = np.rot90(im, j)
                fullData[index] = dat
                fullData[index+1] = np.flipud(dat)
                fullData[index+2] = np.fliplr(dat)

                #Add relevant labels
                lab = np.rot90(ans, j)
                fullScores[index] = lab
                fullScores[index+1] = np.flipud(lab)
                fullScores[index+2] = np.fliplr(lab)

        return fullData, fullScores

    def predict(self,images,threshold=False,thresh=0.9):
        predictions = []
        start = time.time()
        for image in images:
            image = self.normaliseData([image])
            image = np.asarray(image).reshape(1,self.width,
                    self.height,1)
            print("Starting Prediction", time.time()-start)
            predict = self.model.predict(image)
            i = 0
            print("Prediction complete", time.time()-start)
            predictImage = predict
            if threshold:
                predictImage[predictImage>=0.9*thresh] = 1
                predictImage[predictImage<0.9*thresh] = 0
            predictions.append(predictImage[0,:,:,0])
            #predictions.append(predictImage[0,:,:,1])
        return predictions



    def normaliseData(self,images):
        normalised = []
        for image in images:
            normal = image/(np.median(image))
            normalised.append(normal)
        return normalised

    def score(self,images):
        classScores = []
        for image in images:
            image[image>0.0] = 1
            image[image<=0.0] = 0
            cellScores = image
            backGroundScores = (image -1)*-1
            #classScores.append(np.asarray([cellScores,backGroundScores]))
            classScores.append(np.asarray([cellScores]))
        return classScores



    def saveModel(self,path):
        self.model.save(path)

    def loadModel(self,path):
        del self.model
        self.model = load_model(path)

    def loadImage(self,path):
        return np.asarray(imread(path),dtype='uint16')

    def saveImage(self,path,image):
        imsave(path,image)
