import sys
from PyQt5.QtWidgets import QApplication
from Normal import NormalWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NormalWindow()
    sys.exit(app.exec_())
