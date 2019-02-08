import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import numpy as np
from scipy.ndimage import imread
import os

#self.dataDic = {"FolderName":[],"ImageName":[],"Image":[],"ROIs":[],"ROIImages":[]}
class MachineSegmenterApp(QtGui.QWidget):

    def __init__(self,parent=None):
        self.dataDic = {"FolderName":[],"ImageName":[],"Image":[],"ROIs":[],"ROIImages":[]}
        self.app = QtGui.QApplication([])
        QtGui.QWidget.__init__(self,parent)
        self.setUpUI()
        self.setUpMainWidget()
        self.show()

    def setUpMainWidget(self):
        self.setWindowTitle('Machine segmenter app')
        self.resize(800,450)
        self.setLayout(self.UILayout)

    def setUpUI(self):
        self.UILayout = QtGui.QGridLayout()

        #Main Image view
        mainWin = pg.GraphicsLayoutWidget()
        self.mainView = mainWin.addViewBox()
        self.mainView.setAspectLocked(True)
        self.mainImage = pg.ImageItem(border='w')
        self.mainView.addItem(self.mainImage)
        self.mainImage.sigMouseClicked.connect(self.onMainImageClick)


        #ROI Image view
        ROIWin = pg.GraphicsLayoutWidget()
        self.ROIView = ROIWin.addViewBox()
        self.ROIView.setAspectLocked(True)
        self.ROIImage = pg.ImageItem(border='w')
        self.ROIView.addItem(self.ROIImage)

        #Load file button
        loadFilebtn = QtGui.QPushButton("Load file")
        loadFilebtn.clicked.connect(self.onLoadFilePress)

        #Load folder button
        loadFolderbtn = QtGui.QPushButton("Load folder")
        loadFolderbtn.clicked.connect(self.onLoadFolderPress)

        #Image slider
        self.imageSlider = QtGui.QSlider(pg.QtCore.Qt.Horizontal)
        self.imageSlider.setMinimum(1)
        self.imageSlider.setMaximum(1)
        self.imageSlider.valueChanged.connect(self.onImageSliderChange)


        #Add widgets to layouts
        self.UILayout.addWidget(loadFilebtn,0,0)
        self.UILayout.addWidget(loadFolderbtn,0,1)
        self.UILayout.addWidget(mainWin,1,0)
        self.UILayout.addWidget(ROIWin,1,1)
        self.UILayout.addWidget(self.imageSlider,2,0)

    def run(self):
        self.app.exec_()

    def onLoadFilePress(self):
        filePath, filter = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.',filter="*.tiff")
        if filePath != "":
            self.loadImage(filePath)

    def onLoadFolderPress(self):
        folderPath = QtGui.QFileDialog.getExistingDirectory()
        if folderPath != "":
            files = os.listdir(folderPath)
            for f in files:
                filePath = os.path.join(folderPath,f)
                if f.endswith(".tiff"):
                    self.loadImage(filePath)

    def loadImage(self,filepath):
        image = np.asarray(imread(filepath))
        self.mainImage.setImage(image)
        self.dataDic["FolderName"].append(filepath)
        self.dataDic["ImageName"].append(filepath)
        self.dataDic["Image"].append(image)
        self.dataDic["ROIs"].append([])
        self.dataDic["ROIImages"].append([])
        self.imageSlider.setMaximum(len(self.dataDic["Image"]))


        #roi = pg.RectROI([20, 20], [20, 20], pen=(0,9))
        #self.mainView.addItem(roi)
        #roi.sigRegionChangeFinished.connect(self.onROIChange)

    def onROIChange(self):
        subImage = self.sender().getArrayRegion(self.image,self.mainImage)
        self.ROIImage.setImage(subImage)

    def onImageSliderChange(self):
        i =  self.sender().value()
        self.mainImage.setImage(self.dataDic["Image"][i-1])

    def onMainImageClick(self):
        print("CLICK CLICK LCICK")
if __name__ == '__main__':
    from MachineSegmenterApp import MachineSegmenterApp
    MSA1 = MachineSegmenterApp()
    MSA1.run()
    exit()
