from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from eye_model import EYEModel
from datetime import datetime
import pandas as pd
import shutil
import os
import cv2

class CameraWorker(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(21)  #Change to 21 during Deployment
        self.last_frame = None
        self.active = True

    def run(self):
        while self.active:
            ret, frame = self.capture.read()
            if ret:
                self.last_frame = frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_image = QImage(
                    rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                    QImage.Format_RGB888
                ).scaled(250, 250, aspectRatioMode=Qt.KeepAspectRatio)
                self.image_update.emit(qt_image)

    def get_last_frame(self):
        return self.last_frame

    def stop(self):
        self.active = False
        self.quit()
        self.wait()
        self.capture.release()

class EyeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EyeWindow, self).__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        uic.loadUi("../UI/Eye.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)
        self.comboBox_2.currentTextChanged.connect(self.handle_model_switch)

        self.show()
        
        # Global Variables
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.src_directory = os.path.join(root_dir, 'Dataset', 'User_Saved_EYE')
        
        
        # Begin
        self.actionLoad_Sample.triggered.connect(self.load_sample_data)
        self.Data_Table.itemSelectionChanged.connect(self.display_selected_image)
        self.actionExport.triggered.connect(self.export_table_data)
        self.actionLoad.triggered.connect(self.load_user_data)
        self.addButton.clicked.connect(self.add_blank_row)
        self.delButton.clicked.connect(self.delete_row)
        self.Analyze_Button.clicked.connect(self.analyze_data)

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
              
    def load_sample_data(self):
        # Step 1: Locate sample directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sample_dir = os.path.join(root_dir, 'Dataset', 'EYE_sample')
        self.src_directory = sample_dir

        # Step 2: Find CSV
        csv_path = None
        for file in os.listdir(sample_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(sample_dir, file)
                break

        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.critical(self, "Error", "No CSV file found in EYE_sample.")
            return

        df = pd.read_csv(csv_path)

        # Step 3: Prepare the table
        self.Data_Table.setRowCount(0)
        self.Data_Table.setColumnCount(5)

        for i, row in df.iterrows():
            self.Data_Table.insertRow(i)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            # ID
            id_item = QTableWidgetItem(str(i + 1))
            self.Data_Table.setItem(i, 0, id_item)

            # Name
            name_item = QTableWidgetItem(f"Sample {i + 1}")
            self.Data_Table.setItem(i, 1, name_item)

            # Time
            time_item = QTableWidgetItem(timestamp)
            self.Data_Table.setItem(i, 2, time_item)

            # Result (initially blank, read-only)
            result_item = QTableWidgetItem("")
            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, result_item)

            # Take Photo button
            take_photo_btn = QPushButton("Take Photo")
            take_photo_btn.setProperty("row", i)
            take_photo_btn.clicked.connect(self.handle_take_photo)
            self.Data_Table.setCellWidget(i, 4, take_photo_btn)

        self.Data_Table.resizeColumnsToContents()
        
    def handle_take_photo(self):
        button = self.sender()
        row = button.property("row")

        # Prevent photo in sample directory
        if os.path.basename(self.src_directory) == "EYE_sample":
            QMessageBox.warning(self, "Action Not Allowed", "Please export the dataset before taking photos.")
            return

        if not hasattr(self, 'camera_worker') or not self.camera_worker.isRunning():
            self.camera_worker = CameraWorker()
            self.camera_worker.image_update.connect(
                lambda img: self.imageDisplayer.setPixmap(QPixmap.fromImage(img))
            )
            self.camera_worker.start()
            button.setText("Capture")
            self.active_photo_row = row
            self.take_photo_button = button
        else:
            frame = self.camera_worker.get_last_frame()
            self.camera_worker.stop()
            self.imageDisplayer.clear()

            if frame is not None:
                image_id = f"updated{row + 1}"
                filename = f"{image_id}.jpg"
                save_path = os.path.join(self.src_directory, filename)
                cv2.imwrite(save_path, frame)

                # Remove red background on ID
                id_item = self.Data_Table.item(row, 0)
                if id_item:
                    id_item.setBackground(QColor(255, 255, 255))

                # Update image_id and image_path columns
                self.Data_Table.setItem(row, 4, QTableWidgetItem(image_id))

                # Update CSV
                csv_file = next((f for f in os.listdir(self.src_directory) if f.endswith(".csv")), None)
                if csv_file:
                    csv_path = os.path.join(self.src_directory, csv_file)
                    df = pd.read_csv(csv_path)
                    if row < len(df):
                        df.at[row, 'image_id'] = image_id
                        df.to_csv(csv_path, index=False)

                # Display captured image
                pixmap = QPixmap(save_path)
                if not pixmap.isNull():
                    self.imageDisplayer.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio))

            self.take_photo_button.setText("Take Photo")
            self.active_photo_row = None
            
    def display_selected_image(self):
        selected_row = self.Data_Table.currentRow()
        if selected_row == -1 or not self.src_directory:
            return

        # Find CSV file in current src_directory
        csv_file = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
        if not csv_file:
            self.imageDisplayer.setText("CSV not found")
            return

        csv_path = os.path.join(self.src_directory, csv_file)

        try:
            df = pd.read_csv(csv_path)
            if selected_row >= len(df):
                self.imageDisplayer.setText("Row out of bounds")
                return

            image_id = str(df.iloc[selected_row].get("image_id", "")).strip()
            if not image_id:
                self.imageDisplayer.setText("Missing image ID")
                return

            filename = image_id if image_id.endswith(('.jpg', '.png')) else f"{image_id}.jpg"
            image_path = os.path.join(self.src_directory, filename)

            if not os.path.exists(image_path):
                self.imageDisplayer.setText("Image not found")
                return

            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.imageDisplayer.setText("Failed to load image")
                return

            self.imageDisplayer.setPixmap(pixmap.scaled(
                250, 250, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation
            ))

        except Exception as e:
            self.imageDisplayer.setText("Error loading image")
            
    
    def export_table_data(self, folder_path=None):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_EYE')
        os.makedirs(base_dir, exist_ok=True)

        # Step 1: Choose target directory
        if folder_path:
            target_dir = folder_path
        else:
            target_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory", base_dir)
            if not target_dir:
                return

        is_save_mode = os.path.abspath(target_dir) == os.path.abspath(self.src_directory)
        folder_basename = os.path.basename(target_dir)
        csv_path = os.path.join(target_dir, f"{folder_basename}.csv")

        # Step 2: Handle blank project creation
        if self.Data_Table.rowCount() == 0:
            df = pd.DataFrame(columns=["id", "name", "time", "result", "image_id"])
            df.to_csv(csv_path, index=False)
            self.src_directory = target_dir
            QMessageBox.information(self, "Blank Project Created", f"New blank project created at:\n{target_dir}")
            return

        # Step 3: Load original image IDs from source CSV
        source_csv = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
        if not source_csv:
            QMessageBox.critical(self, "Export Error", "No CSV file found in source directory.")
            return
        source_csv_path = os.path.join(self.src_directory, source_csv)
        df_source = pd.read_csv(source_csv_path)

        # Step 4: Export table data
        headers = ["id", "name", "time", "result", "image_id"]
        data = []

        for row in range(self.Data_Table.rowCount()):
            row_id = str(row + 1)
            name = self.Data_Table.item(row, 1).text()
            timestamp = self.Data_Table.item(row, 2).text()
            result = self.Data_Table.item(row, 3).text()

            if row < len(df_source):
                original_image_id = df_source.at[row, 'image_id']
            else:
                original_image_id = ""

            new_image_id = f"eye{row + 1}.jpg"

            if not is_save_mode:
                src_image_path = os.path.join(self.src_directory, original_image_id)
                dst_image_path = os.path.join(target_dir, new_image_id)
                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dst_image_path)

            final_image_id = original_image_id if is_save_mode else new_image_id
            row_data = [row_id, name, timestamp, result, final_image_id]
            data.append(row_data)

        df_export = pd.DataFrame(data, columns=headers)
        df_export.to_csv(csv_path, index=False)

        if not is_save_mode:
            self.src_directory = target_dir
            QMessageBox.information(self, "Export Complete", f"Exported to:\n{target_dir}")
        else:
            QMessageBox.information(self, "Save Complete", f"Saved changes in:\n{self.src_directory}")
            
            
    def load_user_data(self):
        # Step 1: Ask user to select a folder
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_EYE')
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", base_dir)
        if not selected_dir:
            return

        self.src_directory = selected_dir

        # Step 2: Locate the CSV file inside the selected folder
        csv_file = None
        for file in os.listdir(selected_dir):
            if file.endswith(".csv"):
                csv_file = os.path.join(selected_dir, file)
                break

        if not csv_file or not os.path.exists(csv_file):
            QMessageBox.critical(self, "Error", "No CSV file found in selected folder.")
            return

        # Step 3: Load and interpret CSV
        df = pd.read_csv(csv_file)
        self.Data_Table.setRowCount(0)
        self.Data_Table.setColumnCount(5)

        # Determine result column: label or result
        result_col = None
        if "label" in df.columns:
            result_col = "label"
        elif "result" in df.columns:
            result_col = "result"

        for i, row in df.iterrows():
            self.Data_Table.insertRow(i)

            # ID
            self.Data_Table.setItem(i, 0, QTableWidgetItem(str(row.get("id", i + 1))))

            # Name
            self.Data_Table.setItem(i, 1, QTableWidgetItem(str(row.get("name", f"Sample {i + 1}"))))

            # Time
            self.Data_Table.setItem(i, 2, QTableWidgetItem(str(row.get("time", ""))))

            # Result (from 'label' or 'result', read-only)
            if result_col:
                raw_result = row.get(result_col, "")
                result_value = "" if pd.isna(raw_result) else str(raw_result).strip()
            else:
                result_value = ""
            
            result_item = QTableWidgetItem(result_value)
            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, result_item)

            # Take Photo button
            take_photo_btn = QPushButton("Take Photo")
            take_photo_btn.setProperty("row", i)
            take_photo_btn.clicked.connect(self.handle_take_photo)
            self.Data_Table.setCellWidget(i, 4, take_photo_btn)

        self.Data_Table.resizeColumnsToContents()
        
    def add_blank_row(self):
        # If table is empty and no dataset exists, create a new project first
        if self.Data_Table.rowCount() == 0:
            QMessageBox.information(self, "Create Project", "Creating a new blank dataset...")
            self.export_table_data()

            # If export was canceled, skip adding
            if self.src_directory is None or not any(f.endswith('.csv') for f in os.listdir(self.src_directory)):
                QMessageBox.warning(self, "Canceled", "Project creation was canceled.")
                return

        selected = self.Data_Table.currentRow()
        insert_index = self.Data_Table.rowCount() if selected == -1 else selected + 1

        # Generate next ID
        existing_ids = []
        for i in range(self.Data_Table.rowCount()):
            item = self.Data_Table.item(i, 0)
            if item:
                try:
                    existing_ids.append(int(item.text()))
                except ValueError:
                    pass
        next_id = max(existing_ids, default=0) + 1

        # Insert row
        self.Data_Table.insertRow(insert_index)
        id_item = QTableWidgetItem(str(next_id))
        id_item.setBackground(QColor(255, 100, 100))  # Light red
        self.Data_Table.setItem(insert_index, 0, id_item)
        self.Data_Table.setItem(insert_index, 1, QTableWidgetItem(""))  # Name
        self.Data_Table.setItem(insert_index, 2, QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M")))  # Time

        result_item = QTableWidgetItem("")
        result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
        self.Data_Table.setItem(insert_index, 3, result_item)

        take_photo_btn = QPushButton("Take Photo")
        take_photo_btn.setProperty("row", insert_index)
        take_photo_btn.clicked.connect(self.handle_take_photo)
        self.Data_Table.setCellWidget(insert_index, 4, take_photo_btn)

        # Update CSV
        csv_file = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
        if csv_file:
            csv_path = os.path.join(self.src_directory, csv_file)
            df = pd.read_csv(csv_path)

            new_row = {
                "id": next_id,
                "name": "",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "result": "",
                "image_id": ""
            }

            df = pd.concat([df.iloc[:insert_index], pd.DataFrame([new_row]), df.iloc[insert_index:]]).reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            
    def delete_row(self):
        row = self.Data_Table.currentRow()
        if row == -1:
            row = self.Data_Table.rowCount() - 1  # delete last row if none selected

        if row < 0:
            return  # nothing to delete

        # Load and update CSV
        csv_file = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
        if csv_file:
            csv_path = os.path.join(self.src_directory, csv_file)
            df = pd.read_csv(csv_path)

            if row < len(df):
                # Delete associated image
                image_id = df.at[row, 'image_id']
                if isinstance(image_id, str) and image_id.strip():
                    filename = image_id if image_id.endswith(('.jpg', '.png')) else f"{image_id}.jpg"
                    image_path = os.path.join(self.src_directory, filename)
                    if os.path.exists(image_path):
                        os.remove(image_path)

                # Drop the row from CSV and save
                df = df.drop(index=row).reset_index(drop=True)
                df.to_csv(csv_path, index=False)

        # Remove row from UI table
        self.Data_Table.removeRow(row)
        
    def analyze_data(self):
        # Step 1: Force save first
        self.export_table_data(folder_path=self.src_directory)
    
        # Step 2: Run inference
        try:
            model = EYEModel(sample_dir=self.src_directory)
            predictions = model.inference()  # returns full label list
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"Model inference failed:\n{str(e)}")
            return
    
        # Step 3: Update Result column (col 3)
        for i, pred in enumerate(predictions):
            if i < self.Data_Table.rowCount():
                item = QTableWidgetItem(pred)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.Data_Table.setItem(i, 3, item)
    
        QMessageBox.information(self, "Analysis Complete", "Model prediction completed successfully.")
        
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

              
from Skin import SkinWindow
from Normal import NormalWindow
from EMS import EMSWindow
