from PyQt5.QtCore import QRect, Qt
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPainter, QPixmap
from colorizor.swatch_colorizor import RectRegion, colorize_swatch, colorize
import cv2

class TransferPanel(QWidget):

    def __init__(self):
        super(TransferPanel, self).__init__()
        self.ui = loadUi('ui/Transfer.ui', self)
        self.btnLoadImg.clicked.connect(self.loadImg)
        self.btnSaveImg.clicked.connect(self.saveImg)
        self.btnLoadFile.clicked.connect(self.loadRegions)
        self.btnLoadRef.clicked.connect(self.loadRef)
        self.btnColorize.clicked.connect(self.colorize)

        self.imgRegions = []
        self.refRegions = []
        
    def setPixmap(self, img_bgr, is_ref, is_bgr):
        if (is_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr
        
        img_h, img_w, _ = img_bgr.shape
        
        if (img_h > 540):
            ratio = img_h / 540
            img_rgb = cv2.resize(img_rgb, (int(img_w / ratio), 540))
            img_h, img_w, _ = img_rgb.shape
            
            if (is_ref):
                self.refRatio *= ratio
            else:
                self.imgRatio *= ratio
        
        if (img_w > 460):
            ratio = img_w / 460
            img_rgb = cv2.resize(img_rgb, (460, int(img_h / ratio)))
            img_h, img_w, _ = img_rgb.shape

            if (is_ref):
                self.refRatio *= ratio
            else:
                self.imgRatio *= ratio

        bytesPerLine = img_w * 3
        
        img_q = QImage(img_rgb.data, img_w, img_h, bytesPerLine, QImage.Format_RGB888)

        img_margin_x = int((460 - img_w) / 2)
        img_margin_y = int((560 - img_h) / 2)

        if (not is_ref):
            self.img_pixmap  = QPixmap.fromImage(img_q)
            self.ori_imgpixmap = QPixmap.fromImage(img_q)
            self.imgLabel.setGeometry(img_margin_x, img_margin_y, img_w, img_h)
            self.imgLabel.setPixmap(self.img_pixmap)
        
        else:
            self.ref_pixmap  = QPixmap.fromImage(img_q)
            self.ori_refpixmap = QPixmap.fromImage(img_q)
            self.refLabel.setGeometry(img_margin_x, img_margin_y, img_w, img_h)
            self.refLabel.setPixmap(self.ref_pixmap)
        
    def loadImg(self):
        img_path = QFileDialog.getOpenFileName(self, '选择待上色图像', '.', "image file (*.jpg *.png *.bmp)")
        img_path = img_path[0]

        if (len(img_path) == 0):
            return

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.img_gray = img

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if (img is None):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "无效的图像文件")
            imgNotLoadMsg.exec()
            return

        self.img = img
        self.img_colorized = None
        self.setPixmap(img, False, True)

        del self.imgRegions[:]
        del self.refRegions[:]
    
    def loadRef(self):
        img_path = QFileDialog.getOpenFileName(self, '选择参考图像', '.', "image file (*.jpg *.png *.bmp)")
        img_path = img_path[0]

        if (len(img_path) == 0):
            return

        img = cv2.imread(img_path)
        if (img is None):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "无效的图像文件")
            imgNotLoadMsg.exec()
            return

        self.ref = img
        self.img_colorized = None
        self.setPixmap(img, True, True)

        del self.imgRegions[:]
        del self.refRegions[:]

    def loadRegions(self):
        img_path = QFileDialog.getOpenFileName(self, '选择区域文件', '.', "")
        img_path = img_path[0]

        if (len(img_path) == 0):
            return

        status = 0

        del self.imgRegions[:]
        del self.refRegions[:]

        with open(img_path) as f:
            for line in f:
                if (line.strip() == "REF"):
                    status = 2
                elif (line.strip() == "IMG"):
                    status = 1
                else:
                    tmp = []
                    for i in line.split():
                        try:
                            tmp.append(int(i))
                        except ValueError:
                            del self.imgRegions[:]
                            del self.refRegions[:]
                            invalidMsg = QMessageBox(QMessageBox.Critical, "文件无效", "不合规的区域标定文件")
                            invalidMsg.exec()
                            return
                    if (len(tmp) != 4):
                        del self.imgRegions[:]
                        del self.refRegions[:]
                        invalidMsg = QMessageBox(QMessageBox.Critical, "文件无效", "不合规的区域标定文件")
                        invalidMsg.exec()
                        return
                    if (status == 1):
                        self.imgRegions.append(RectRegion(tmp[0], tmp[1], tmp[2], tmp[3]))
                    elif (status == 2):
                        self.refRegions.append(RectRegion(tmp[0], tmp[1], tmp[2], tmp[3]))

        loaddMsg = QMessageBox(QMessageBox.Information, "加载成功", "区域文件加载成功")
        loaddMsg.exec()                  

    def colorize(self):
        if (not hasattr(self, 'img')):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待上色图像")
            imgNotLoadMsg.exec()
            return

        if (not hasattr(self, 'ref')):
            refNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待参考图像")
            refNotLoadMsg.exec()
            return

        out = None

        if (len(self.refRegions) == 0):
            _, out = colorize_swatch(self.img_gray, self.ref)
        else:
            out = colorize(self.img_gray, self.ref, self.imgRegions, self.refRegions)
        
        self.img_colorized = out
        self.setPixmap(self.img_colorized, False, True)

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