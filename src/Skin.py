from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem, QComboBox, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from skin_model import SKINModel
from datetime import datetime
import pandas as pd
import shutil
import os
import cv2

class CameraWorker(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        self.capture = cv2.VideoCapture(21) #Use 21 on Deployment
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_image = QImage(
                    image.data, image.shape[1], image.shape[0],
                    QImage.Format_RGB888
                )
                scaled = qt_image.scaled(250, 250, Qt.KeepAspectRatio)
                self.image_update.emit(scaled)
                self.last_frame = frame

    def stop(self):
        self.running = False
        if hasattr(self, 'capture'):
            self.capture.release()
        self.quit()

    def get_last_frame(self):
        return getattr(self, 'last_frame', None)    

class SkinWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(SkinWindow, self).__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        uic.loadUi("../UI/Skin.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)
        self.comboBox_2.currentTextChanged.connect(self.handle_model_switch)

        self.show()
        
        # Global Variables
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.src_directory = os.path.join(root_dir, 'Dataset', 'User_Saved_SKIN')
        
        ## Begin
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
        if text == "Eye Disease":
            self.new_window = EyeWindow()
            self.new_window.show()
            self.close()
        elif text == "Stroke Prediction":
            self.new_window = NormalWindow()
            self.new_window.show()
            self.close()
            
            
    def load_sample_data(self):
        # Load SKIN sample metadata
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sample_dir = os.path.join(root_dir, 'Dataset', 'SKIN_sample')
        self.src_directory = sample_dir
        csv_path = os.path.join(sample_dir, 'SKIN_sample.csv')
        df = pd.read_csv(csv_path)

        self.Data_Table.setRowCount(0)
        self.Data_Table.setColumnCount(9)  # Now includes "Take Photo" button

        for i, row in df.iterrows():
            self.Data_Table.insertRow(i)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            # ID, Name, Time
            self.Data_Table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.Data_Table.setItem(i, 1, QTableWidgetItem(f"Sample {i + 1}"))
            self.Data_Table.setItem(i, 2, QTableWidgetItem(timestamp))

            # Result (empty, read-only)
            result_item = QTableWidgetItem("")
            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, result_item)

            # Dx Type (ComboBox)
            dx_cb = QComboBox()
            dx_cb.addItems(["histo", "follow_up", "consensus"])
            dx_cb.setCurrentText(str(row.get("dx_type", "")).lower())
            self.Data_Table.setCellWidget(i, 4, dx_cb)

            # Age
            self.Data_Table.setItem(i, 5, QTableWidgetItem(str(row.get("age", ""))))

            # Sex (ComboBox)
            sex_cb = QComboBox()
            sex_cb.addItems(["male", "female"])
            sex_cb.setCurrentText(str(row.get("sex", "")).lower())
            self.Data_Table.setCellWidget(i, 6, sex_cb)

            # Localization (ComboBox)
            loc_cb = QComboBox()
            loc_cb.addItems([
                'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
                'genital', 'hand', 'lower extremity', 'neck', 'scalp',
                'trunk', 'unknown', 'upper extremity'
            ])
            loc_cb.setCurrentText(str(row.get("localization", "")).lower())
            self.Data_Table.setCellWidget(i, 7, loc_cb)

            # Take Photo button
            take_photo_btn = QPushButton("Take Photo")
            take_photo_btn.setProperty("row", i)
            take_photo_btn.clicked.connect(self.handle_take_photo)
            self.Data_Table.setCellWidget(i, 8, take_photo_btn)

        self.Data_Table.resizeColumnsToContents()

        
    def display_selected_image(self):
        selected_row = self.Data_Table.currentRow()
        if selected_row == -1 or not self.src_directory:
            return  # No selection or source not set

        # Path to current CSV in source directory
        csv_path = os.path.join(self.src_directory, 'SKIN_sample.csv')
        # If exported, use the renamed CSV file
        for file in os.listdir(self.src_directory):
            if file.endswith('.csv'):
                csv_path = os.path.join(self.src_directory, file)
                break

        try:
            df = pd.read_csv(csv_path)
            image_id = df.iloc[selected_row]['image_id']
            image_path = os.path.join(self.src_directory, image_id)
            
            #print(image_path)

            if os.path.exists(image_path):
                self.imageDisplayer.clear()
                self.imageDisplayer.setText("Image not found")
                return

            pixmap = QPixmap(image_path) 
            if pixmap.isNull():
                self.imageDisplayer.setText("Please Take a Photo")
                return

            self.imageDisplayer.setPixmap(pixmap.scaled(
                250, 250, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation
            ))

        except Exception:
            self.imageDisplayer.setText("Error loading image")
            
    def handle_take_photo(self):
        button = self.sender()
        row = button.property("row")

        # Prevent photo capture on protected sample data
        if self.src_directory and os.path.basename(self.src_directory) == "SKIN_sample":
            QMessageBox.warning(self, "Action Not Allowed", "You must export the dataset before taking photos.")
            return

        # Start camera stream
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
            # Stop stream and capture image
            frame = self.camera_worker.get_last_frame()
            self.camera_worker.stop()
            self.imageDisplayer.clear()

            if frame is not None:
                # Use updated<ID> to avoid cache issues
                image_id = f"updated{row + 1}"
                filename = f"{image_id}.jpg"
                save_path = os.path.join(self.src_directory, filename)

                # Save captured image
                cv2.imwrite(save_path, frame)
                
                # Remove red background on ID cell once photo is captured
                id_item = self.Data_Table.item(row, 0)
                if id_item:
                    id_item.setBackground(QColor(255, 255, 255))  # Reset to white

                # Refresh display immediately
                pixmap = QPixmap(save_path)
                if not pixmap.isNull():
                    self.imageDisplayer.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio))

                # Update CSV: image_id and image_path
                csv_name = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
                if csv_name:
                    csv_path = os.path.join(self.src_directory, csv_name)
                    df = pd.read_csv(csv_path)

                    if row < len(df):
                        df.at[row, 'image_id'] = image_id
                        df.at[row, 'image_path'] = filename
                        df.to_csv(csv_path, index=False)

            # Reset button state
            self.take_photo_button.setText("Take Photo")
            self.active_photo_row = None
            
    def export_table_data(self, folder_path=None):
        # Mapping for dx label conversion
        dx_label_map = {
            'nv': 'Melanocytic Nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign Keratosis-like Lesions',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratoses',
            'vasc': 'Vascular Lesions',
            'df': 'Dermatofibroma'
        }
        full_to_abbr = {v: k for k, v in dx_label_map.items()}

        # Step 1: Select or create export folder

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_SKIN')
        os.makedirs(base_dir, exist_ok=True)
        target_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", base_dir
        )
        if not target_dir:
            return

        is_save_mode = os.path.abspath(target_dir) == os.path.abspath(self.src_directory)

        # Determine CSV path
        if is_save_mode:
            # Use existing CSV file in the same directory
            csv_path = None
            for file in os.listdir(self.src_directory):
                if file.endswith('.csv'):
                    csv_path = os.path.join(self.src_directory, file)
                    break
            if not csv_path:
                QMessageBox.critical(self, "Save Error", "No CSV file found in current dataset directory.")
                return
        else:
            # Create new CSV in new folder
            folder_basename = os.path.basename(target_dir)
            csv_path = os.path.join(target_dir, f"{folder_basename}.csv")
            
        # Handle completely blank table: just create empty CSV
        if self.Data_Table.rowCount() == 0:
            df = pd.DataFrame(columns=["image_id", "dx", "dx_type", "age", "sex", "localization", "image_path"])
            df.to_csv(csv_path, index=False)

            # Set the current project directory
            self.src_directory = target_dir

            QMessageBox.information(self, "Blank Project Created", f"New blank project created at:\n{target_dir}")
            return

        # Step 2: Read original image IDs from source CSV
        source_csv_path = None
        for file in os.listdir(self.src_directory):
            if file.endswith('.csv'):
                source_csv_path = os.path.join(self.src_directory, file)
                break

        # Only check for existing CSV if table is not empty
        if self.Data_Table.rowCount() > 0:
            if not source_csv_path or not os.path.exists(source_csv_path):
                QMessageBox.critical(self, "Export Error", "No valid CSV file found in source directory.")
                return

        try:
            df_source = pd.read_csv(source_csv_path)
            source_image_ids = df_source['image_id'].tolist()
        except Exception:
            QMessageBox.critical(self, "Export Error", "Failed to read CSV file in source directory.")
            return

        # Step 3: Export table contents
        headers = ["image_id", "dx", "dx_type", "age", "sex", "localization", "image_path"]
        data = []

        for row in range(self.Data_Table.rowCount()):
            # Safely get original image info
            image_id = f"skin{row + 1}"
            filename = f"{image_id}.jpg"

            if row < len(source_image_ids):
                image_id = source_image_ids[row]
                filename = f"{image_id}.jpg"

            # Map result full label to abbreviation
            result_item = self.Data_Table.item(row, 3)
            full_label = result_item.text().strip() if result_item else ""
            dx = full_to_abbr.get(full_label, "")

            dx_type_cb = self.Data_Table.cellWidget(row, 4)
            sex_cb = self.Data_Table.cellWidget(row, 6)
            loc_cb = self.Data_Table.cellWidget(row, 7)

            row_data = [
                image_id,
                dx,
                dx_type_cb.currentText().lower(),
                self.Data_Table.item(row, 5).text(),
                sex_cb.currentText().lower(),
                loc_cb.currentText().lower(),
                filename
            ]
            data.append(row_data)

            # If exporting to a new directory, copy the images too
            if not is_save_mode:
                src_image_path = os.path.join(self.src_directory, filename)
                dst_image_path = os.path.join(target_dir, filename)
                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dst_image_path)

        # Write new CSV (overwrite if exists)
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(csv_path, index=False)

        # If exported to a new folder, update src_directory to match
        if not is_save_mode:
            self.src_directory = target_dir
            QMessageBox.information(self, "Export Complete", f"Exported to:\n{target_dir}")
        else:
            QMessageBox.information(self, "Save Complete", f"Saved changes in:\n{self.src_directory}")

        
    def load_user_data(self):
        # 1. Ask user for directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_SKIN')
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", base_dir
        )
        if not selected_dir:
            return

        self.src_directory = selected_dir

        # 2. Locate CSV file in selected directory
        csv_file = None
        for file in os.listdir(selected_dir):
            if file.endswith(".csv"):
                csv_file = os.path.join(selected_dir, file)
                break

        if not csv_file or not os.path.exists(csv_file):
            QMessageBox.critical(self, "Error", "No CSV file found in selected folder.")
            return

        # 3. Mapping from short dx to full name
        dx_label_map = {
            'nv': 'Melanocytic Nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign Keratosis-like Lesions',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratoses',
            'vasc': 'Vascular Lesions',
            'df': 'Dermatofibroma'
        }

        # 4. Load CSV
        df = pd.read_csv(csv_file)
        self.Data_Table.setRowCount(0)
        self.Data_Table.setColumnCount(9)

        for i, row in df.iterrows():
            self.Data_Table.insertRow(i)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            # ID, Name, Time
            self.Data_Table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.Data_Table.setItem(i, 1, QTableWidgetItem(f"Sample {i + 1}"))
            self.Data_Table.setItem(i, 2, QTableWidgetItem(timestamp))

            # Result: map dx to full label
            dx_code_raw = row.get("dx", "")
            dx_code = str(dx_code_raw).strip() if pd.notna(dx_code_raw) else ""
            full_label = dx_label_map.get(dx_code, "")
            result_item = QTableWidgetItem(full_label)
            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, result_item)

            # Dx Type ComboBox
            dx_cb = QComboBox()
            dx_cb.addItems(["histo", "follow_up", "consensus"])
            dx_cb.setCurrentText(str(row.get("dx_type", "")).lower())
            self.Data_Table.setCellWidget(i, 4, dx_cb)

            # Age
            self.Data_Table.setItem(i, 5, QTableWidgetItem(str(row.get("age", ""))))

            # Sex ComboBox
            sex_cb = QComboBox()
            sex_cb.addItems(["male", "female"])
            sex_cb.setCurrentText(str(row.get("sex", "")).lower())
            self.Data_Table.setCellWidget(i, 6, sex_cb)

            # Localization ComboBox
            loc_cb = QComboBox()
            loc_cb.addItems([
                'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
                'genital', 'hand', 'lower extremity', 'neck', 'scalp',
                'trunk', 'unknown', 'upper extremity'
            ])
            loc_cb.setCurrentText(str(row.get("localization", "")).lower())
            self.Data_Table.setCellWidget(i, 7, loc_cb)

            # Take Photo button
            take_photo_btn = QPushButton("Take Photo")
            take_photo_btn.setProperty("row", i)
            take_photo_btn.clicked.connect(self.handle_take_photo)
            self.Data_Table.setCellWidget(i, 8, take_photo_btn)

        self.Data_Table.resizeColumnsToContents()
        
    def add_blank_row(self):
        # If table is empty and no dataset exists, create a new project first
        if self.Data_Table.rowCount() == 0:
            QMessageBox.information(self, "Create Project", "Creating a new blank dataset...")

            # Trigger export with empty table
            self.export_table_data()

            # Check again: if export was canceled, skip adding row
            if self.src_directory is None or not any(f.endswith(".csv") for f in os.listdir(self.src_directory)):
                QMessageBox.warning(self, "Canceled", "Project creation was canceled.")
                return
        
        
        selected = self.Data_Table.currentRow()
        insert_index = self.Data_Table.rowCount()
    
        self.Data_Table.insertRow(insert_index)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
        # Placeholder image ID (will be blank until photo is taken)
        image_id = f"updated{insert_index + 1}"
        image_filename = f"{image_id}.jpg"
    
        # ID and Name
        id_item = QTableWidgetItem(str(insert_index + 1))
        id_item.setBackground(QColor(255, 100, 100))  # Light red to indicate missing photo
        self.Data_Table.setItem(insert_index, 0, id_item)               # ID
        self.Data_Table.setItem(insert_index, 1, QTableWidgetItem(f"Sample {insert_index + 1}"))  # Name left blank
        self.Data_Table.setItem(insert_index, 2, QTableWidgetItem(timestamp))  # Time
    
        # Result (blank, read-only)
        result_item = QTableWidgetItem("")
        result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
        self.Data_Table.setItem(insert_index, 3, result_item)
    
        # Dx Type (ComboBox)
        dx_cb = QComboBox()
        dx_cb.addItems(["histo", "follow_up", "consensus"])
        self.Data_Table.setCellWidget(insert_index, 4, dx_cb)
    
        # Age
        self.Data_Table.setItem(insert_index, 5, QTableWidgetItem(""))
    
        # Sex (ComboBox)
        sex_cb = QComboBox()
        sex_cb.addItems(["male", "female"])
        self.Data_Table.setCellWidget(insert_index, 6, sex_cb)
    
        # Localization (ComboBox)
        loc_cb = QComboBox()
        loc_cb.addItems([
            'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
            'genital', 'hand', 'lower extremity', 'neck', 'scalp',
            'trunk', 'unknown', 'upper extremity'
        ])
        self.Data_Table.setCellWidget(insert_index, 7, loc_cb)
    
        # Take Photo button
        take_photo_btn = QPushButton("Take Photo")
        take_photo_btn.setProperty("row", insert_index)
        take_photo_btn.clicked.connect(self.handle_take_photo)
        self.Data_Table.setCellWidget(insert_index, 8, take_photo_btn)
    
        # Append a new row to the CSV
        csv_file = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
        if csv_file:
            csv_path = os.path.join(self.src_directory, csv_file)
            df = pd.read_csv(csv_path)
    
            new_row = {
                "image_id": image_id,
                "dx": "",
                "dx_type": "",
                "age": "",
                "sex": "",
                "localization": "",
                "image_path": image_filename
            }
            df = df._append(new_row, ignore_index=True)
            df.to_csv(csv_path, index=False)
            
            
    def delete_row(self):
        row_to_delete = self.Data_Table.currentRow()
        if row_to_delete == -1:
            row_to_delete = self.Data_Table.rowCount() - 1

        if row_to_delete < 0:
            return  # Nothing to delete

        # Remove from CSV
        csv_file = next((f for f in os.listdir(self.src_directory) if f.endswith('.csv')), None)
        if csv_file:
            csv_path = os.path.join(self.src_directory, csv_file)
            df = pd.read_csv(csv_path)

            if row_to_delete < len(df):
                image_file = df.at[row_to_delete, "image_path"]
                image_path = os.path.join(self.src_directory, image_file)
                if os.path.exists(image_path):
                    os.remove(image_path)

                df = df.drop(index=row_to_delete).reset_index(drop=True)
                df.to_csv(csv_path, index=False)

        # Remove from UI table
        self.Data_Table.removeRow(row_to_delete)
        
    
    def analyze_data(self):
        # Step 1: Force save/export
        self.export_table_data(folder_path=self.src_directory)

        # Step 2: Instantiate model with proper sample_dir
        try:
            model = SKINModel(sample_dir=self.src_directory)
            predictions = model.inference()  # List of dx codes: ['mel', 'nv', ...]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model inference failed:\n{str(e)}")
            return

        # Step 3: Map dx codes to full names
        dx_label_map = {
            'nv': 'Melanocytic Nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign Keratosis-like Lesions',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratoses',
            'vasc': 'Vascular Lesions',
            'df': 'Dermatofibroma'
        }

        for i, pred in enumerate(predictions):
            full_label = dx_label_map.get(pred, "Unknown")
            item = QTableWidgetItem(full_label)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, item)
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
    
    
    
from Eye import EyeWindow
from Normal import NormalWindow
from EMS import EMSWindow
