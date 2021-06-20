from PyQt5.QtCore import QRect
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from colorizor.sketch_colorizor import colorize
import cv2

class SketchColorizerPanel(QWidget):

    def __init__(self):
        super(SketchColorizerPanel, self).__init__()
        self.ui = loadUi('ui\\Sketch.ui', self)
        self.btnLoadImg.clicked.connect(self.loadImg)
        self.btnLoadSketch.clicked.connect(self.loadSketch)
        self.btnColorize.clicked.connect(self.colorize)
        self.btnSaveImg.clicked.connect(self.saveImg)

    def setPixmap(self, img_bgr, is_sketch, is_bgr):
        if (is_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr
        
        img_h, img_w, _ = img_bgr.shape
        
        if (img_h > 650):
            ratio = img_h / 650
            img_rgb = cv2.resize(img_rgb, (int(img_w / ratio), 650))
            img_h, img_w, _ = img_rgb.shape
        
        if (img_w > 1200):
            ratio = img_w / 1200
            img_rgb = cv2.resize(img_rgb, (1200, int(img_h / ratio)))
            img_h, img_w, _ = img_rgb.shapes

        bytesPerLine = img_w * 3
        
        img_q = QImage(img_rgb.data, img_w, img_h, bytesPerLine, QImage.Format_RGB888)
        
        self.window_w = max(481, img_w + 40)
        self.window_h = img_h + 65 + 51

        img_margin_x = int((self.window_w - img_w) / 2)

        if (not is_sketch):
            self.img_pixmap  = QPixmap.fromImage(img_q)
            self.imgLabel.setGeometry(img_margin_x, 20, img_w, img_h)
            self.imgLabel.setPixmap(self.img_pixmap)
        
        else:
            self.sketch_pixmap  = QPixmap.fromImage(img_q)
            self.sketchLabel.setGeometry(img_margin_x, 20, img_w, img_h)
            self.sketchLabel.setPixmap(self.sketch_pixmap)

        self.tabWidget.setGeometry(0, 0, self.window_w, self.window_h - 51)

        self.buttonFrame.setGeometry(QRect(0, self.window_h - 51, self.window_w, 51))
        
        self.resize(self.window_w, self.window_h)

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
    
    def loadSketch(self):
        sketch_path = QFileDialog.getOpenFileName(self, '选择引导图像', '.', "image file (*.jpg *.png *.bmp)")
        sketch_path = sketch_path[0]

        if (len(sketch_path) == 0):
            return

        img = cv2.imread(sketch_path)
        if (img is None):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "无效的图像文件")
            imgNotLoadMsg.exec()
            return

        self.sketch = img
        self.img_colorized = None
        self.setPixmap(img, True, True)

    def colorize(self):
        if (not hasattr(self, 'img')):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待上色图像")
            imgNotLoadMsg.exec()
            return

        if (not hasattr(self, 'sketch')):
            sketchNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载引导图像")
            sketchNotLoadMsg.exec()
            return

        img_colorized = colorize(self.img, self.sketch)
        self.setPixmap(img_colorized, False, True)
        self.img_colorized = img_colorized

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