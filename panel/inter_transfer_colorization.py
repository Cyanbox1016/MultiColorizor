from PyQt5.uic import *
from PyQt5.QtWidgets import *

class InterTransferPanel(QWidget):

    def __init__(self):
        super(InterTransferPanel, self).__init__()
        self.ui = loadUi('ui\\InteractiveTransfer.ui', self)