from PyQt5.QtCore import QRect, Qt
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPainter, QPixmap
from colorizor.swatch_colorizor import RectRegion, colorize_swatch, colorize
import cv2

class InterTransferPanel(QWidget):

    def __init__(self):
        super(InterTransferPanel, self).__init__()
        self.ui = loadUi('ui/InteractiveTransfer.ui', self)
        self.btnLoadImg.clicked.connect(self.loadImg)
        self.btnLoadRef.clicked.connect(self.loadRef)
        self.btnAddRegion.clicked.connect(self.addRefRegion)
        self.btnRemoveRegion.clicked.connect(self.removeRegion)
        self.btnColorize.clicked.connect(self.colorize)
        self.btnSaveImg.clicked.connect(self.saveImg)      
        
        self.refRegions = []
        self.imgRegions = []

        self.onAdd = False
        self.refSwatchStatus = 0
        self.imgSwatchStatus = 0

        self.img_begin_x = 0
        self.img_begin_y = 0
        self.img_end_x = 0
        self.img_end_y = 0

        self.ref_begin_x = 0
        self.ref_begin_y = 0
        self.ref_end_x = 0
        self.ref_end_y = 0

        self.imgRatio = 1.
        self.refRatio = 1.
        
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

    def isInImage(self, x:int, y:int, isRef:bool):
        groupBoxGeo = None
        LabelGeo = None
        if (not isRef):
            groupBoxGeo = self.groupBox_2.geometry()
            LabelGeo = self.imgLabel.geometry()
        else:
            groupBoxGeo = self.groupBox.geometry()
            LabelGeo = self.refLabel.geometry()
        
        x_ = x - groupBoxGeo.left()
        y_ = y - groupBoxGeo.top()

        if (LabelGeo.contains(x_, y_)):
            return True
        else:
            return False

    def transWindowCoordToImgCoord(self, x:int, y:int, isRef:bool):
        groupBoxGeo = None
        LabelGeo = None

        if (not isRef):
            groupBoxGeo = self.groupBox_2.geometry()
            LabelGeo = self.imgLabel.geometry()
        else:
            groupBoxGeo = self.groupBox.geometry()
            LabelGeo = self.refLabel.geometry()
        
        x_ = x - LabelGeo.left() - groupBoxGeo.left()
        y_ = y - LabelGeo.top() - groupBoxGeo.top()

        if (x_ < 0):
            x_ = 0
        elif (x_ >= LabelGeo.width()):
            x_ = LabelGeo.width() - 1

        if (y_ < 0):
            y_ = 0
        elif (y_ >= LabelGeo.height()):
            y_ = LabelGeo.height() - 1

        return x_, y_
    
    def transPixCoordToImgCoord(self, x: int, y: int, isRef:bool):
        x_ = None
        y_ = None
        
        if (isRef):
            x_ = round(x * self.refRatio)
            y_ = round(y * self.refRatio)
        else:
            x_ = round(x * self.imgRatio)
            y_ = round(y * self.imgRatio)
        
        return int(x_), int(y_)

    def addRefRegion(self):
        if (not hasattr(self, 'img')):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待上色图像")
            imgNotLoadMsg.exec()
            return

        if (not hasattr(self, 'ref')):
            refNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待参考图像")
            refNotLoadMsg.exec()
            return
        
        self.onAdd = True
    
    def mousePressEvent(self, QMouseEvent):
        if (self.onAdd):
            if (self.isInImage(QMouseEvent.x(), QMouseEvent.y(), False) and self.imgSwatchStatus == 0):
                self.img_begin_x, self.img_begin_y = self.transWindowCoordToImgCoord(QMouseEvent.x(), QMouseEvent.y(), False)
                self.imgSwatchStatus = 1
            elif (self.isInImage(QMouseEvent.x(), QMouseEvent.y(), True) and self.refSwatchStatus == 0):
                self.ref_begin_x, self.ref_begin_y = self.transWindowCoordToImgCoord(QMouseEvent.x(), QMouseEvent.y(), True)
                self.refSwatchStatus = 1
        
    def mouseReleaseEvent(self, QMouseEvent):
        if (self.onAdd):
            if (self.imgSwatchStatus == 1):
                self.img_end_x, self.img_end_y = self.transWindowCoordToImgCoord(QMouseEvent.x(), QMouseEvent.y(), False)
                painter = QPainter(self.img_pixmap)

                if (self.img_begin_x > self.img_end_x):
                    self.img_begin_x, self.img_end_x = self.img_end_x, self.img_begin_x
                if (self.img_begin_y > self.img_end_y):
                    self.img_begin_y, self.img_end_y = self.img_end_y, self.img_begin_y

                painter.drawRect(self.img_begin_x, self.img_begin_y, self.img_end_x - self.img_begin_x, self.img_end_y - self.img_begin_y)
                self.imgLabel.setPixmap(self.img_pixmap)
                self.imgSwatchStatus = 2

            elif (self.refSwatchStatus == 1):
                self.ref_end_x, self.ref_end_y = self.transWindowCoordToImgCoord(QMouseEvent.x(), QMouseEvent.y(), True)

                if (self.ref_begin_x > self.ref_end_x):
                    self.ref_begin_x, self.ref_end_x = self.ref_end_x, self.ref_begin_x
                if (self.ref_begin_y > self.ref_end_y):
                    self.ref_begin_y, self.ref_end_y = self.ref_end_y, self.ref_begin_y

                painter = QPainter(self.ref_pixmap)
                painter.drawRect(self.ref_begin_x, self.ref_begin_y, self.ref_end_x - self.ref_begin_x, self.ref_end_y - self.ref_begin_y)
                self.refLabel.setPixmap(self.ref_pixmap)
                self.refSwatchStatus = 2
        
        if (self.refSwatchStatus == 2 and self.imgSwatchStatus == 2):
            self.onAdd = False
            self.refSwatchStatus = 0
            self.imgSwatchStatus = 0

            ibx, iby = self.transPixCoordToImgCoord(self.img_begin_x, self.img_begin_y, False)
            iex, iey = self.transPixCoordToImgCoord(self.img_end_x, self.img_end_y, False)

            rbx, rby = self.transPixCoordToImgCoord(self.ref_begin_x, self.ref_begin_y, True)
            rex, rey = self.transPixCoordToImgCoord(self.ref_end_x, self.ref_end_y, True)

            self.imgRegions.append(RectRegion(ibx, iby, iex - ibx, iey - iby))
            self.refRegions.append(RectRegion(rbx, rby, rex - rbx, rey - rby))

    def removeRegion(self):
        self.onAdd = False
        self.refSwatchStatus = 0
        self.imgSwatchStatus = 0

        del self.imgRegions[:]
        del self.refRegions[:]

        self.img_pixmap = self.ori_imgpixmap.copy()
        self.ref_pixmap = self.ori_refpixmap.copy()

        self.refLabel.setPixmap(self.ref_pixmap)
        self.imgLabel.setPixmap(self.img_pixmap)

    def colorize(self):
        if (not hasattr(self, 'img')):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待上色图像")
            imgNotLoadMsg.exec()
            return

        if (not hasattr(self, 'ref')):
            refNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待参考图像")
            refNotLoadMsg.exec()
            return

        if (self.onAdd):
            regionErrorMsg = QMessageBox(QMessageBox.Critical, "错误", "有未完成的参考区域")
            regionErrorMsg.exec()
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