from PyQt5.QtCore import QRect
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
import colorizor.cnn_colorizor
import cv2

class DeepImgPanel(QWidget):

    def __init__(self):
        super(DeepImgPanel, self).__init__()
        self.ui = loadUi('ui/DeepImageColorization.ui', self)
        self.btnLoadImg.clicked.connect(self.loadImg)
        self.btnColorize.clicked.connect(self.colorize)
        self.btnSaveImg.clicked.connect(self.saveImg)

    def setPixmap(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
        self.img_pixmap  = QPixmap.fromImage(img_q)

        self.window_w = max(400, img_w + 50)
        self.window_h = img_h + 25 + 85

        img_margin_x = int((self.window_w - img_w) / 2)

        self.imgLabel.setGeometry(img_margin_x, 25, img_w, img_h)
        self.imgLabel.setPixmap(self.img_pixmap)

        self.buttonFrame.setGeometry(QRect(0, self.window_h - 85, self.window_w, 85))
        
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
        self.setPixmap(img)

    def colorize(self):
        if (not hasattr(self, 'img')):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待上色图像")
            imgNotLoadMsg.exec()
            return
        
        ckpt_path = QFileDialog.getOpenFileName(self, '选择预训练模型', 'colorizor\\model')
        ckpt_path = ckpt_path[0]

        if (len(ckpt_path) == 0):
            return

        confirmMsg = QMessageBox(QMessageBox.Question, "上色", "开始使用该模型进行图像上色？", QMessageBox.Yes|QMessageBox.No)
        retval = confirmMsg.exec_()
        
        if (retval == 65536):
            return
        elif (retval == 16384):
            img_colorized = None

            try:
                img_colorized = colorizor.cnn_colorizor.colorize(self.img, ckpt_path, "colorizor\\model\\pts_in_hull.npy")
            except BaseException:
                colorizedFailedMsg = QMessageBox(QMessageBox.Critical, "失败", "上色失败！请检查图像和模型文件以及运行环境！")
                colorizedFailedMsg.exec()
                return    
            
            self.img_colorized = img_colorized
            self.setPixmap(img_colorized)

            return img_colorized

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
