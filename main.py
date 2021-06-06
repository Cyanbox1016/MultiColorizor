# import sys
# from PyQt5.uic import *
# from PyQt5.QtWidgets import *

# class MainMenu(QWidget):
#     def __init__(self):
#         super(MainMenu, self).__init__()
#         self.ui = loadUi('ui\\MainMenu.ui', self)
#         self.btnTransferColor.clicked.connect(self.close)

#     def close(self):
#         print("hello owlr")


# def main():
#     app = QApplication(sys.argv)
#     ex = MainMenu()
#     ex.show()
#     sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()

from colorizor.cnn_colorizor_2 import colorize
import cv2

img = cv2.imread("g1.jpg")
ref = cv2.imread("k2.bmp")

img = colorize(img, "colorizor\\model\\model.ckpt-119000", "colorizor\\model\\pts_in_hull.npy")
cv2.imshow("title", img)
cv2.waitKey(0)

