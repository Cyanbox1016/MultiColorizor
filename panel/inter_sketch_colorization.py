from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
from colorizor.sketch_colorizor import colorize

class SketchToolBar(QWidget):

    class SketchDrawerPanel(QWidget):
        def __init__(self):
            super().__init__()
            self.ui = loadUi('ui/InteractiveSketch.ui', self)
            self.x_last = 0;
            self.y_last = 0;
            
            self.sketchMap = None
            self.ori_sketchMap = None
            self.pen = QPen()
            self.pen.setColor(Qt.white)
        
        def setPen(self, pen):
            self.pen = pen

        def mousePressEvent(self, QMouseEvent):
            self.pressed = 1
            self.x_last = QMouseEvent.x()
            self.y_last = QMouseEvent.y()
        
        def mouseReleaseEvent(self, QMouseEvent):
            self.pressed = 0

        def mouseMoveEvent(self, QMouseEvent):
            if (self.pressed == 1 and self.sketchMap is not None):
                x_ = QMouseEvent.x()
                y_ = QMouseEvent.y()
                painter = QPainter(self.sketchMap)
                painter.setPen(self.pen)
                painter.setRenderHint(QPainter.Antialiasing, True);
                painter.drawLine(self.x_last, self.y_last, x_, y_)
                self.x_last = x_
                self.y_last = y_
                self.label_2.setPixmap(self.sketchMap)

    def __init__(self):
        super().__init__()
        self.ui = loadUi("ui/ColorToolBar.ui", self)
        # 
        self.main = self.SketchDrawerPanel()

        self.btnColorSelect.clicked.connect(self.selectColor)
        self.btnLoadImg.clicked.connect(self.loadImg)
        self.btnClear.clicked.connect(self.clear)
        self.btnColorize.clicked.connect(self.colorize)
        self.btnSaveImg.clicked.connect(self.saveImg)
        
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
            self.main.pen.setColor(color)

    def clear(self):
        if (self.main.ori_sketchMap is not None):
            self.main.sketchMap = self.main.ori_sketchMap.copy()
            self.main.label_2.setPixmap(self.main.sketchMap)
    
    def changeTip(self):
        if (self.tipSpinBox.value() != self.tip):
            self.tip = self.tipSpinBox.value()
            self.tipSlider.setValue(self.tip)
        elif (self.tipSlider.value() != self.tip):
            self.tip = self.tipSlider.value()
            self.tipSpinBox.setValue(self.tip)
        self.main.pen.setWidth(self.tip)
    
    def setPixmap(self, img_bgr, is_sketch, is_bgr):
        if (is_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr
        
        img_h, img_w, _ = img_bgr.shape
        
        if (img_h > 800):
            ratio = img_h / 800
            img_rgb = cv2.resize(img_rgb, (int(img_w / ratio), 800))
            img_h, img_w, _ = img_rgb.shape
        
        if (img_w > 1200):
            ratio = img_w / 1200
            img_rgb = cv2.resize(img_rgb, (1200, int(img_h / ratio)))
            img_h, img_w, _ = img_rgb.shapes

        bytesPerLine = img_w * 3
        
        img_q = QImage(img_rgb.data, img_w, img_h, bytesPerLine, QImage.Format_RGB888)
        
        self.main.window_w = img_w
        self.main.window_h = img_h

        self.main.label.setPixmap(QPixmap.fromImage(img_q))
        self.main.label.setGeometry(0, 0, img_w, img_h)
        
        self.main.sketchMap = QPixmap(img_w, img_h)
        self.main.sketchMap.fill(Qt.transparent)
        self.main.label_2.setGeometry(0, 0, img_w, img_h)

        self.main.ori_sketchMap = self.main.sketchMap.copy()
        self.main.label_2.setPixmap(self.main.sketchMap)

        self.main.resize(img_w, img_h)
        
        self.setGeometry(1600, 405, 194, 270)
        self.main.show()
    
    def loadImg(self):
        img_path = QFileDialog.getOpenFileName(self, '选择待上色图像', '.', "image file (*.jpg *.png *.bmp)")
        img_path = img_path[0]

        if (len(img_path) == 0):
            return

        img = cv2.imread(img_path)
        if (img is None):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "无效的图像文件")
            imgNotLoadMsg.exec()
            return

        self.img = img
        self.img_colorized = None

        self.setPixmap(img, False, True)
    
    def colorize(self):
        self.main.sketchMap.save("tmp.bmp")
        sketch = cv2.imread("tmp.bmp")
        img = self.img
        
        self.img_colorized = colorize(img, sketch)
        self.setPixmap(self.img_colorized, False, True)

        os.remove("tmp.bmp")
    
    def saveImg(self):
        if (not hasattr(self, "img_colorized") or self.img_colorized is None):
            notColorizedMsg = QMessageBox(QMessageBox.Critical, "保存图像无效", "还没有生成上色后的图像")
            notColorizedMsg.exec()
            return
        
        img_path = QFileDialog.getSaveFileName(self, '选择图像存储路径', '.', "JPEG image(*.jpg);;Bitmap(*.bmp);;PNG img(*.png)")
        img_path = img_path[0]
        
        if (len(img_path) == 0):
            return
        
        cv2.imwrite(img_path, self.img_colorized)
        savedMsg = QMessageBox(QMessageBox.Information, "保存成功", "图像已保存")
        savedMsg.exec()