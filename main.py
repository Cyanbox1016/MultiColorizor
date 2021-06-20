import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from PyQt5.uic import *
from PyQt5.QtWidgets import *
from panel.main_menu import MainMenu

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec_())
