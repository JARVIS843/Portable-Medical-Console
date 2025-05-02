import sys
import subprocess
from PyQt5 import QtWidgets, uic

class EyeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EyeWindow, self).__init__()
        uic.loadUi("Eye.ui", self)

        try:
            self.comboBox_1.currentTextChanged.connect(self.handle_mode_switch)
        except AttributeError:
            pass

        self.show()

    def handle_mode_switch(self, text):
        if text == "EMS":
            subprocess.Popen([sys.executable, "EMS.py"])
            self.close()
        elif text == "Normal":
            subprocess.Popen([sys.executable, "Normal.py"])
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EyeWindow()
    sys.exit(app.exec_())
