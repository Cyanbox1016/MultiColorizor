import sys
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from panel.deep_img_colorization import DeepImgPanel
from panel.inter_sketch_colorization import SketchDrawerPanel
from panel.inter_transfer_colorization import InterTransferPanel
from panel.sketch_colorization import SketchColorizerPanel
from panel.video_colorization import VideoColorizePanel

class MainMenu(QWidget):
    def __init__(self):
        super(MainMenu, self).__init__()
        self.ui = loadUi('ui/MainMenu.ui', self)
        self.btnSketch.clicked.connect(self.openSketchPanel)
        self.btnDeepLearning.clicked.connect(self.openDeepImgPanel)
        self.btnSketchInteractive.clicked.connect(self.openSketchInterPanel)
        self.btnTransferColorInteractive.clicked.connect(self.openInteractiveTransferPanel)
        self.btnDeepLearning_2.clicked.connect(self.openVideoColorizePanel)

    def openSketchPanel(self):
        sketchPanel = SketchColorizerPanel()
        sketchPanel.show()

    def openDeepImgPanel(self):
        deepImgPanel = DeepImgPanel()
        deepImgPanel.show()
    
    def openSketchInterPanel(self):
        sketchDrawer = SketchDrawerPanel()
        sketchDrawer.run()

    def openInteractiveTransferPanel(self):
        interTransferPanel = InterTransferPanel()
        interTransferPanel.show()

    def openVideoColorizePanel(self):
        videoPanel = VideoColorizePanel()
        videoPanel.show()