from PyQt5.uic import *
from PyQt5.QtWidgets import *

class SketchDrawerPanel(QWidget):

    class SketchToolBar(QWidget):
        
        def __init__(self, main):
            super().__init__()
            self.ui = loadUi("ui/ColorToolBar.ui", self)
            self.main = main
            self.btnColorSelect.clicked.connect(self.selectColor)
            self.penTypeBox.currentIndexChanged.connect(self.selectPenType)
            
            self.tipSlider.setMinimum(1)
            self.tipSlider.setMaximum(50)
            self.tipSpinBox.setMinimum(1)
            self.tipSpinBox.setMaximum(50)
            self.tip = 1
            self.tipSpinBox.valueChanged.connect(self.changeTip)
            self.tipSlider.valueChanged.connect(self.changeTip)

        def selectColor(self):
            color = QColorDialog.getColor()
            if (color.isValid()):
                self.color = color

        def selectPenType(self):
            self.penType = self.penTypeBox.currentIndex()
        
        def changeTip(self):
            if (self.tipSpinBox.value() != self.tip):
                self.tip = self.tipSpinBox.value()
                self.tipSlider.setValue(self.tip)
            elif (self.tipSlider.value() != self.tip):
                self.tip = self.tipSlider.value()
                self.tipSpinBox.setValue(self.tip)
            
            
    def __init__(self):
        super(SketchDrawerPanel, self)
        self.toolBar = self.SketchToolBar(self)
    
    def run(self):
        self.toolBar.show()