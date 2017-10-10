from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import sys
import numpy as np

class MachineSegmenter:

    def __init__(self):
        self.model = None
        self.data = None
        self.answers = None
        self.scores = None
        self.rfSize = None

    def defineModel(self, num_classes=2, kernel_size=3, pool_size=2,
            conv_depth_1=5, conv_depth_2=3, hidden_size=100, rfSize=21):
        self.rfSize = rfSize
        inp = Input(shape=(rfSize,rfSize,1))
        conv_1 = Convolution2D(conv_depth_1,(kernel_size,kernel_size),
                padding='same',activation='relu')(inp)
        conv_2 = Convolution2D(conv_depth_1,(kernel_size,kernel_size),
                padding='same',activation='relu')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv_2)
        conv_3 = Convolution2D(conv_depth_2,(kernel_size,kernel_size),
                padding='same',activation='relu')(pool_1)
        conv_3 = Convolution2D(conv_depth_2,(kernel_size,kernel_size),
                padding='same',activation='relu')(conv_3)
        conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                padding='same', activation='relu')(conv_3)
        pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
        flat = Flatten()(pool_2)
        hidden = Dense(hidden_size, activation='relu')(flat)
        out = Dense(num_classes, activation='softmax')(hidden)
        self.model = Model(inputs=inp, outputs=out)

    def compileModel(self):
        if self.model != None:
            self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        else:
            print('Error: Tried to compile model which was undefined')
            sys.exit(0)

    def loadTrainingData(self,data,answers,asImage=True):
        if asImage:
            data = self.slice(data,self.rfSize)
            data = self.normaliseData(data)
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
            data = np.asarray(self.data).reshape(len(self.data),self.rfSize,self.rfSize,1)
            scores = np.asarray(self.scores)
            self.model.fit(data, scores, batch_size=batch_size,
                    epochs=num_epochs, verbose =1)
        else:
            print("Error data or answers not initialised!")
            sys.exit(0)

    def slice(self,images,rfSize):
        sliced = []
        for image in images:
            width,height = image.shape
            padded = np.pad(image,rfSize,'constant')
            for x in range(rfSize,width+rfSize):
                for y in range(rfSize,height+rfSize):
                    sliced.append(padded[x:x+rfSize,y:y+rfSize])
        return sliced

    def normaliseData(self,images):
        normalised = []
        for image in images:
            normal = image - np.mean(image)
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

    def predict(self,images):
        predictions = []
        for image in images:
            predictImage = np.zeros(np.shape(image))
            image = self.slice([image],self.rfSize)
            image = self.normaliseData(image)
            image = np.asarray(image).reshape(len(image),self.rfSize,
                    self.rfSize,1)
            predict = self.model.predict(image)
            i = 0
            for x in range(predictImage.shape[0]):
                for y in range(predictImage.shape[1]):
                    predictImage[x][y] = predict[i][0]
                    i+=1
            predictions.append(predictImage)
        return predictions
