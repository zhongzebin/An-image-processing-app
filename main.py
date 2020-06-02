from PyQt5 import QtWidgets
from ui_connect import Ui_connect

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_connect()
    ui.show()
    app.exec_()