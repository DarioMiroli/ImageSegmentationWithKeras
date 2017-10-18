from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
import numpy as np

'''def squareSlice(x, size = 21):
    dims = np.full(2, size)
    shape = np.array(x.shape*2)
    shape[x.ndim:] = dims
    shape[:x.ndim] -= dims - 1
    strides = np.array(x.strides*2)
    squares = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    squares = squares.reshape((-1,dims[0],dims[1]))
    return squares

l = np.ones((100,100))
c = 1
for i in range(l.shape[0]):
    for j in range(l.shape[1]):
        l[i,j] = c
        c = c+1

a = np.zeros((0,2,2))

a = np.append(a, squareSlice(l, size = 2), axis = 0)

print(l.strides)
print (a[10])'''

M1 = MachineSegmenter()
M1.defineModel(conv_depth_1 = 1)
M1.compileModel()
D1 = DataGenerator(1334)
trainingData, trainingAnswers = D1.generateData(images=1,recNo=20)
#D1.displayData(trainingData,1)
#D1.displayData(trainingAnswers,1)
#Train model
M1.loadTrainingData(trainingData,trainingAnswers)
M1.trainModel(batch_size=256)
validationData , _ = D1.generateData(images=1,recNo = 20)
validationPredicts = M1.predict(validationData)
D1.displayData(validationData, delay=20)
D1.displayData(validationPredicts, delay=20)
