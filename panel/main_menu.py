import sys
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from panel.deep_img_colorization import DeepImgPanel

class MainMenu(QWidget):
    def __init__(self):
        super(MainMenu, self).__init__()
        self.ui = loadUi('ui\\MainMenu.ui', self)
        self.btnDeepLearning.clicked.connect(self.openDeepImgPanel)

    def openDeepImgPanel(self):
        deepImgPanel = DeepImgPanel()
        deepImgPanel.show()