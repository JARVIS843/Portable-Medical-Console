from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QPushButton, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import os
import csv
import cv2
import pandas as pd
from datetime import datetime


class CameraThread(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(21)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            scaled = img.scaled(400, 400, Qt.KeepAspectRatio)
            self.image_update.emit(scaled)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class EMSWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EMSWindow, self).__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        uic.loadUi("../UI/EMS.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)

        self.show()
        
        # Begin
        self.Data_Table = self.findChild(QtWidgets.QTableWidget, "Data_Table")
        self.addButton = self.findChild(QtWidgets.QPushButton, "addButton")
        self.imageDisplayer = self.findChild(QtWidgets.QLabel, "imageDisplayer")
        self.Data_Table.itemSelectionChanged.connect(self.display_selected_image)
        
        self.capturing = False
        self.cap_thread = None
        self.current_camera_row = None

        self.addButton.clicked.connect(self.add_row)
        self.timers = {}
        self.tq_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'TQs'))
        self.csv_path = os.path.join(self.tq_dir, 'TQs.csv')

    def handle_mode_switch(self, text):
        if text == "Normal":
            self.new_window = NormalWindow()
            self.new_window.show()
            self.close()
    
    def add_row(self):
        row_index = self.Data_Table.rowCount()
        self.Data_Table.insertRow(row_index)

        # ID
        id_item = QTableWidgetItem(str(row_index + 1))
        id_item.setBackground(QColor(0, 255, 0))  # green
        self.Data_Table.setItem(row_index, 0, id_item)

        # Name (empty)
        self.Data_Table.setItem(row_index, 1, QTableWidgetItem(""))

        # Start timer at 15:00
        time_item = QTableWidgetItem("15:00")
        time_item.setFlags(time_item.flags() & ~Qt.ItemIsEditable)
        self.Data_Table.setItem(row_index, 2, time_item)

        # Start countdown timer (per-row)
        countdown = {"minutes": 15, "seconds": 0}
        timer = QTimer(self)
        timer.timeout.connect(lambda ri=row_index, cd=countdown: self.update_timer(ri, cd))
        timer.start(1000)
        self.timers[row_index] = timer

        # Take Photo Button
        photo_button = QPushButton("Take Photo")
        photo_button.setProperty("row", row_index)
        photo_button.clicked.connect(self.handle_take_photo)
        self.Data_Table.setCellWidget(row_index, 3, photo_button)
        
        # Resize Column Widths
        #self.Data_Table.resizeColumnsToContents()

        # Write metadata to CSV
        image_id = ""  # initially empty
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row_index + 1, "", now, image_id])
            
            
    def update_timer(self, row, countdown):
        if row >= self.Data_Table.rowCount():
            return  # row might have been deleted
    
        if countdown["minutes"] == 0 and countdown["seconds"] == 0:
            self.timers[row].stop()
            id_item = self.Data_Table.item(row, 0)
            if id_item:
                id_item.setBackground(QColor(255, 0, 0))  # red
            return
    
        if countdown["seconds"] == 0:
            countdown["minutes"] -= 1
            countdown["seconds"] = 59
        else:
            countdown["seconds"] -= 1
    
        # Format time as MM:SS
        formatted = f"{countdown['minutes']:02}:{countdown['seconds']:02}"
        time_item = self.Data_Table.item(row, 2)
        if time_item:
            time_item.setText(formatted)
            
            
    def display_selected_image(self):
        row = self.Data_Table.currentRow()
        if row == -1:
            return

        csv_path = self.csv_path
        df = pd.read_csv(csv_path)
        if row >= len(df):
            self.imageDisplayer.setText("No image available")
            return

        image_id = df.iloc[row]["image_id"]
        if not isinstance(image_id, str) or not image_id.strip():
            self.imageDisplayer.setText("No photo taken")
            return

        img_path = os.path.join(self.tq_dir, image_id)
        if not os.path.exists(img_path):
            self.imageDisplayer.setText("Image not found")
            return

        pixmap = QPixmap(img_path)
        self.imageDisplayer.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        
    def handle_take_photo(self):
        button = self.sender()
        row = button.property("row")

        if not self.capturing:
            # Start streaming
            self.capturing = True
            self.current_camera_row = row
            button.setText("Capture Photo")

            self.cap_thread = CameraThread()
            self.cap_thread.image_update.connect(self.update_preview_from_qimage)
            self.cap_thread.start()

        else:
            # Stop and capture
            self.capturing = False
            button.setText("Take Photo")
            self.cap_thread.stop()

            # Save image
            image_id = f"TQ{row + 1}.jpg"
            save_path = os.path.join(self.tq_dir, image_id)

            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(save_path, frame)

            # Update CSV
            df = pd.read_csv(self.csv_path)
            if row < len(df):
                df.at[row, "image_id"] = image_id
                df.to_csv(self.csv_path, index=False)

            # Refresh display
            self.display_selected_image()

    def update_preview_from_qimage(self, qimg):
        self.imageDisplayer.setPixmap(QPixmap.fromImage(qimg))
        
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            
            

from Normal import NormalWindow
