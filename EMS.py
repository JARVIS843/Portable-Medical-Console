import sys
import subprocess
from PyQt5 import QtWidgets, uic

class EMSWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EMSWindow, self).__init__()
        uic.loadUi("EMS.ui", self)

        # Only listen to comboBox_1
        try:
            self.comboBox_1.currentTextChanged.connect(self.handle_mode_switch)
        except AttributeError:
            pass

        self.show()

    def handle_mode_switch(self, text):
        if text == "Normal":
            subprocess.Popen([sys.executable, "Normal.py"])
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EMSWindow()
    sys.exit(app.exec_())
