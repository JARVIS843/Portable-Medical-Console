from PyQt5 import QtWidgets, uic

class EyeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EyeWindow, self).__init__()
        uic.loadUi("../UI/Eye.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)
        self.comboBox_2.currentTextChanged.connect(self.handle_model_switch)

        self.show()

    def handle_mode_switch(self, text):
        if text == "EMS":
            self.new_window = EMSWindow()
            self.new_window.show()
            self.close()

    def handle_model_switch(self, text):
        if text == "Stroke Prediction":
            self.new_window = NormalWindow()
            self.new_window.show()
            self.close()
            
        elif text == "Skin Disease":
              self.new_window = SkinWindow()
              self.new_window.show()
              self.close()
              
from Skin import SkinWindow
from Normal import NormalWindow
from EMS import EMSWindow
