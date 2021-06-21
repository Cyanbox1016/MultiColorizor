from PyQt5.QtCore import QRect
from PyQt5.uic import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from colorizor.video_colorizor import colorize

import cv2
import os

class VideoColorizePanel(QWidget):

    def __init__(self):
        super(VideoColorizePanel, self).__init__()
        self.ui = loadUi('ui/VideoColorization.ui', self)
        self.btnLoadVideo.clicked.connect(self.loadVideo)
        self.btnColorize.clicked.connect(self.colorize)
    
    def loadVideo(self):
        vid_path = QFileDialog.getOpenFileName(self, '选择待上色图像', '.', "video file (*.avi *.mp4 *.mpg *.mpeg)")
        vid_path = vid_path[0]
        vid_name = os.path.basename(vid_path)

        if (len(vid_path) == 0):
            return

        self.label.setText("视频：" + vid_name)
        self.vid_path = vid_path

    def colorize(self):

        if (not hasattr(self, 'vid_path')):
            imgNotLoadMsg = QMessageBox(QMessageBox.Critical, "错误", "尚未加载待上色图像")
            imgNotLoadMsg.exec()
            return

        confirmMsg = QMessageBox(QMessageBox.Question, "上色", "视频上色等待时间较长，确认开始吗？", QMessageBox.Yes|QMessageBox.No)
        retval = confirmMsg.exec_()

        if (retval == 65536):
            return

        colorize(self.vid_path)

        savedMsg = QMessageBox(QMessageBox.Information, "上色成功", "视频已保存到out.mp4")
        savedMsg.exec()
        
    
    
    