from DataGenerator import DataGenerator
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy as np
# Generate squares and data
D1 = DataGenerator(1)
answers = D1.generateSquares(1)
data = D1.generateData(answers)
#trim and normalise data
receptiveFieldSize = 21
slicedAnswers = D1.slice(answers,receptiveFieldSize)
slicedData = D1.slice(data,receptiveFieldSize)
normalised = D1.normaliseData(slicedData)
classScores = D1.score(slicedAnswers)
#Machine learning part

classArray = np.zeros((500,500))

i = 0
for x in range(500):
    for y in range(500):
        classArray[x,y]=classScores[i][0]
        i+=1
D1.displayData([classArray],delay=2)
D1.displayData(data,delay=2)
normalised = np.asarray(normalised).reshape(len(slicedAnswers),receptiveFieldSize,receptiveFieldSize,1)
classScores = np.asarray(classScores)
#Displays
#D1.displayData(answers,delay=3)
#D1.displayData(data,delay=3)
#D1.displayData(slicedAnswers[250:-1:5000],delay=0.001)
#D1.displayData(normalised[250:-1:5000],delay=0.0007)


#neural net stuff here

num_classes = 2
batch_size = 12
num_epochs = 1
kernel_size = 3
pool_size =2
conv_depth_1 = 1
conv_depth_2 = 1
hidden_size = 100

inp = Input(shape=(receptiveFieldSize,receptiveFieldSize,1))
conv_1 = Convolution2D(conv_depth_1,(kernel_size,kernel_size),padding='same',activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1,(kernel_size,kernel_size),padding='same',activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv_2)
conv_3 = Convolution2D(conv_depth_2,(kernel_size,kernel_size),padding='same',activation='relu')(pool_1)
conv_3 = Convolution2D(conv_depth_2,(kernel_size,kernel_size),padding='same',activation='relu')(conv_3)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
flat = Flatten()(pool_2)
hidden = Dense(hidden_size, activation='relu')(flat)
out = Dense(num_classes, activation='softmax')(hidden)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('\n***********Training********    \n',len(slicedAnswers))
model.fit(normalised, classScores, batch_size=batch_size, epochs=num_epochs, verbose =1, validation_split=0.1)
print('\n*********Finished Training******** \n')
model.save("./modelTest.md5")
modelPredictions = np.zeros((500,500))
i = 0
predictions = model.predict(normalised)
for x in range(500):
    for y in range(500):
        modelPredictions[x,y]= predictions[i][0]
        i+=1
D1.displayData([modelPredictions],delay=30)
