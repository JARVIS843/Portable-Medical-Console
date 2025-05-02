import sys
import subprocess
from PyQt5 import QtWidgets, uic

class NormalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(NormalWindow, self).__init__()
        uic.loadUi("Normal.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)
        self.comboBox_2.currentTextChanged.connect(self.handle_model_switch)

        self.show()

    def handle_mode_switch(self, text):
        if text == "EMS":
            subprocess.Popen([sys.executable, "EMS.py"])
            self.close()

    def handle_model_switch(self, text):
        if text == "Eye Disease":
            subprocess.Popen([sys.executable, "Eye.py"])
            self.close()
        elif text == "Skin Disease":
            subprocess.Popen([sys.executable, "Skin.py"])
            self.close()
