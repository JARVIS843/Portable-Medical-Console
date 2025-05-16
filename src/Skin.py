from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem, QComboBox, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox

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
        self.capture = cv2.VideoCapture(0) #Use 21 on Deployment
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
        uic.loadUi("../UI/Skin.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)
        self.comboBox_2.currentTextChanged.connect(self.handle_model_switch)

        self.show()
        
        # Global Variables
        self.src_directory = None
        
        ## Begin
        self.actionLoad_Sample.triggered.connect(self.load_sample_data)
        self.Data_Table.itemSelectionChanged.connect(self.display_selected_image)
        self.actionExport.triggered.connect(self.export_table_data)
        #self.actionLoad.triggered.connect(self.load_user_data)
        #self.addButton.clicked.connect(self.add_blank_row)
        #self.delButton.clicked.connect(self.delete_row)
        #self.Analyze_Button.clicked.connect(self.analyze_data)

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
            return  # No selection or no source directory set

        # Resolve CSV path from current source directory
        csv_path = None
        for file in os.listdir(self.src_directory):
            if file.endswith('.csv'):
                csv_path = os.path.join(self.src_directory, file)
                break

        if not csv_path or not os.path.exists(csv_path):
            self.imageDisplayer.setText("CSV file not found")
            return

        try:
            df = pd.read_csv(csv_path)

            # Determine which column to use for the image filename
            if 'image_path' in df.columns:
                image_filename = df.iloc[selected_row]['image_path']
            elif 'image_id' in df.columns:
                image_id = df.iloc[selected_row]['image_id']
                image_filename = f"{image_id}.jpg"
            else:
                self.imageDisplayer.setText("Missing image info")
                return

            image_path = os.path.join(self.src_directory, image_filename)
        

            if not os.path.exists(image_path):
                self.imageDisplayer.clear()
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
            
    def handle_take_photo(self):
        button = self.sender()
        row = button.property("row")
        
        # Prevent photo capture on protected sample data
        if self.src_directory and os.path.basename(self.src_directory) == "SKIN_sample":
            QMessageBox.warning(self, "Action Not Allowed", "You must export the dataset before taking photos.")
            return

        if not hasattr(self, 'camera_worker') or not self.camera_worker.isRunning():
            # Start camera stream
            self.camera_worker = CameraWorker()
            self.camera_worker.image_update.connect(
                lambda img: self.imageDisplayer.setPixmap(QPixmap.fromImage(img))
            )
            self.camera_worker.start()
            button.setText("Capture")
            self.active_photo_row = row
            self.take_photo_button = button
        else:
            # Capture and replace image
            frame = self.camera_worker.get_last_frame()
            self.camera_worker.stop()
            self.imageDisplayer.clear()

            if frame is not None:
                image_id = f"skin{row + 1}"
                filename = f"{image_id}.jpg"
                save_path = os.path.join(self.src_directory, filename)
                cv2.imwrite(save_path, frame)

                # Refresh image display
                pixmap = QPixmap(save_path)
                if not pixmap.isNull():
                    self.imageDisplayer.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio))

                # Update CSV
                csv_path = os.path.join(self.src_directory, f"{os.path.basename(self.src_directory)}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if row < len(df):
                        df.at[row, 'image_id'] = image_id
                        df.at[row, 'image_path'] = filename
                        df.to_csv(csv_path, index=False)

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

        # Step 1: Select/create export folder
        if folder_path is None:
            target_dir = folder_path
        else:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            base_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_SKIN')
            os.makedirs(base_dir, exist_ok=True)

            folder_name, _ = QFileDialog.getSaveFileName(
                self,
                "Create Folder for Export",
                os.path.join(base_dir, "NewSkinDataset"),
                "All Files (*)"
            )
            if not folder_name:
                return
            target_dir = folder_name
            os.makedirs(target_dir, exist_ok=True)

        folder_basename = os.path.basename(target_dir)
        csv_path = os.path.join(target_dir, f"{folder_basename}.csv")

        # Step 2: Read image IDs from source CSV
        source_csv_path = os.path.join(self.src_directory, 'SKIN_sample.csv')
        try:
            df_source = pd.read_csv(source_csv_path)
            source_image_ids = df_source['image_id'].tolist()
        except Exception:
            QMessageBox.critical(self, "Export Error", "Failed to read source SKIN_sample.csv.")
            return

        # Step 3: Export data row by row
        headers = ["image_id", "dx", "dx_type", "age", "sex", "localization", "image_path"]
        data = []

        for row in range(self.Data_Table.rowCount()):
            # Actual image_id from source CSV (in order)
            if row >= len(source_image_ids):
                QMessageBox.warning(self, "Export Warning", "Row index out of bounds for image ID.")
                continue

            original_image_id = source_image_ids[row]  # e.g., image00005
            original_filename = f"{original_image_id}.jpg"

            # New image ID and file name
            new_image_id = f"skin{row + 1}"
            new_filename = f"{new_image_id}.jpg"

            # Copy image from original to new folder under new name
            src_image_path = os.path.join(self.src_directory, original_filename)
            dst_image_path = os.path.join(target_dir, new_filename)
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)

            # Extract and map result value
            result_item = self.Data_Table.item(row, 3)
            full_label = result_item.text().strip() if result_item else ""
            dx = full_to_abbr.get(full_label, "")

            dx_type_cb = self.Data_Table.cellWidget(row, 4)
            sex_cb = self.Data_Table.cellWidget(row, 6)
            loc_cb = self.Data_Table.cellWidget(row, 7)

            row_data = [
                new_image_id,
                dx,
                dx_type_cb.currentText().lower(),
                self.Data_Table.item(row, 5).text(),
                sex_cb.currentText().lower(),
                loc_cb.currentText().lower(),
                new_filename
            ]
            data.append(row_data)

        df = pd.DataFrame(data, columns=headers)
        df.to_csv(csv_path, index=False)

        # Set new working directory for future export/inference
        self.src_directory = target_dir

        QMessageBox.information(self, "Export Complete", f"Exported to:\n{target_dir}")
    



from Eye import EyeWindow
from Normal import NormalWindow
from EMS import EMSWindow
