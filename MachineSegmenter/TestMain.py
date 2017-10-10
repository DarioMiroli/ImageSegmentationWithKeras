from MachineSegmenter import MachineSegmenter
from DataGenerator import DataGenerator
#Generate Data
D1 =DataGenerator(1.0)
data, answers = D1.generateData(images=1,recNo=10)
D1.displayData(data,1)
D1.displayData(answers,1)

#Machine learning
M1 = MachineSegmenter()
M1.defineModel()
M1.compileModel()
M1.loadTrainingData(data,answers)
M1.trainModel()

#Analysis
predictions = M1.predict([data[0]])
D1.displayData([data[0]],delay=3)
D1.displayData(predictions,delay=10)
