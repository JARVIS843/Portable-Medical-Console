from PyQt5 import QtWidgets, uic

class EMSWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EMSWindow, self).__init__()
        uic.loadUi("../UI/EMS.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)

        self.show()

    def handle_mode_switch(self, text):
        if text == "Normal":
            self.new_window = NormalWindow()
            self.new_window.show()
            self.close()

from Normal import NormalWindow
