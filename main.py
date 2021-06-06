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

from colorizor.sketch_colorizor import colorize
import cv2

img = cv2.imread("out.jpg")
ref = cv2.imread("k2.bmp")

colorize(img, ref)
