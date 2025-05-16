from PyQt5 import QtWidgets, uic
import pandas as pd
from datetime import datetime

from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QTableWidgetItem, QComboBox
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

from sp_model import SPModel
import os

class NormalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(NormalWindow, self).__init__()
        uic.loadUi("../UI/Normal.ui", self)

        self.comboBox.currentTextChanged.connect(self.handle_mode_switch)
        self.comboBox_2.currentTextChanged.connect(self.handle_model_switch)

        self.show()
        
        
        ## Begin
        
        self.actionLoad_Sample.triggered.connect(self.load_sample_data)
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
        elif text == "Skin Disease":
            self.new_window = SkinWindow()
            self.new_window.show()
            self.close()
            
    def load_sample_data(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dataset_path = os.path.join(root_dir, 'Dataset', 'SP_sample.csv')
        df = pd.read_csv(dataset_path)

        self.Data_Table.setRowCount(0)
        self.Data_Table.setColumnCount(14)

        for i, row in df.iterrows():
            self.Data_Table.insertRow(i)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Set fixed text fields by column index
            self.Data_Table.setItem(i, 0, QTableWidgetItem(str(i + 1)))                      # ID
            self.Data_Table.setItem(i, 1, QTableWidgetItem(f"Sample {i + 1}"))              # Name
            self.Data_Table.setItem(i, 2, QTableWidgetItem(timestamp))                      # Time (Modified)

            #self.Data_Table.setItem(i, 3, result_item)
            self.Data_Table.setItem(i, 3, QTableWidgetItem(str('')))
            self.Data_Table.setItem(i, 5, QTableWidgetItem(str(row.get("age", ""))))        # Age
            self.Data_Table.setItem(i, 6, QTableWidgetItem(str(row.get("hypertension", ""))))
            self.Data_Table.setItem(i, 7, QTableWidgetItem(str(row.get("heart_disease", ""))))
            self.Data_Table.setItem(i, 11, QTableWidgetItem(str(row.get("avg_glucose_level", ""))))
            self.Data_Table.setItem(i, 12, QTableWidgetItem(str(row.get("bmi", ""))))

            # Categorical inputs as QComboBoxes
            combo_data = [
                (4, ["Male", "Female", "Other"], row.get("gender", "").lower()),
                (8, ["Yes", "No"], row.get("ever_married", "")),
                (9, ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], row.get("work_type", "")),
                (10, ["Urban", "Rural"], row.get("Residence_type", "")),
                (13, ["never smoked", "formerly smoked", "smokes", "Unknown", "other"], row.get("smoking_status", ""))
            ]

            for col_idx, options, current_value in combo_data:
                combo = QComboBox()
                combo.addItems(options)
                if current_value in options:
                    combo.setCurrentText(current_value)
                self.Data_Table.setCellWidget(i, col_idx, combo)

        self.Data_Table.resizeColumnsToContents()
    
    def export_table_data(self, file_path=None):
        
        if file_path is not None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            save_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_SP')
            os.makedirs(save_dir, exist_ok=True)
            default_path = os.path.join(save_dir, 'user_saved.csv')
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File", default_path, "CSV Files (*.csv)")
            if not file_path:
                return  # Cancelled manually

        headers = [
            "id", "name", "time_modified",
            "gender", "age", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
            "stroke"  # final column
        ]

        data = []
        for row in range(self.Data_Table.rowCount()):
            row_data = []

            row_data.append(self.Data_Table.item(row, 0).text())  # ID
            row_data.append(self.Data_Table.item(row, 1).text())  # Name
            row_data.append(self.Data_Table.item(row, 2).text())  # Time Modified

            # Categorical fields via ComboBox
            gender = self.Data_Table.cellWidget(row, 4).currentText()
            married = self.Data_Table.cellWidget(row, 8).currentText()
            work = self.Data_Table.cellWidget(row, 9).currentText()
            residence = self.Data_Table.cellWidget(row, 10).currentText()
            smoke = self.Data_Table.cellWidget(row, 13).currentText()

            row_data.extend([
                gender,
                self.Data_Table.item(row, 5).text(),  # Age
                self.Data_Table.item(row, 6).text(),  # Hypertension
                self.Data_Table.item(row, 7).text(),  # Heart Disease
                married,
                work,
                residence,
                self.Data_Table.item(row, 11).text(),  # Glucose
                self.Data_Table.item(row, 12).text(),  # BMI
                smoke
            ])

            # Result column → stroke (Likely/Unlikely → 1/0)
            result_text = self.Data_Table.item(row, 3).text()
            stroke_val = "1" if result_text == "Likely" else "0" if result_text == "Unlikely" else ""
            row_data.append(stroke_val)

            data.append(row_data)

        df = pd.DataFrame(data, columns=headers)
        df.to_csv(file_path, index=False)
    
    def load_user_data(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_SP')
        os.makedirs(default_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Saved Dataset", default_dir, "CSV Files (*.csv)"
        )
        
        if not file_path:
            return

        df = pd.read_csv(file_path)

        self.Data_Table.setRowCount(0)
        self.Data_Table.setColumnCount(14)

        for i, row in df.iterrows():
            self.Data_Table.insertRow(i)

            # Set ID, Name, Time
            self.Data_Table.setItem(i, 0, QTableWidgetItem(str(row.get("id", ""))))
            self.Data_Table.setItem(i, 1, QTableWidgetItem(str(row.get("name", ""))))
            self.Data_Table.setItem(i, 2, QTableWidgetItem(str(row.get("time_modified", ""))))

            # Result → display Likely/Unlikely, color, and lock
            stroke = str(row.get("stroke", "")).strip()
            if stroke == "1":
                result_item = QTableWidgetItem("Likely")
                result_item.setBackground(QColor(255, 100, 100))  # Red
            elif stroke == "0":
                result_item = QTableWidgetItem("Unlikely")
                result_item.setBackground(QColor(100, 255, 100))  # Green
            else:
                result_item = QTableWidgetItem("")

            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, result_item)

            # Gender (ComboBox)
            gender_cb = QComboBox()
            gender_cb.addItems(["Male", "Female", "Other"])
            gender_cb.setCurrentText(str(row.get("gender", "")).lower())
            self.Data_Table.setCellWidget(i, 4, gender_cb)

            # Other text fields
            self.Data_Table.setItem(i, 5, QTableWidgetItem(str(row.get("age", ""))))
            self.Data_Table.setItem(i, 6, QTableWidgetItem(str(row.get("hypertension", ""))))
            self.Data_Table.setItem(i, 7, QTableWidgetItem(str(row.get("heart_disease", ""))))

            # Ever Married
            married_cb = QComboBox()
            married_cb.addItems(["Yes", "No"])
            married_cb.setCurrentText(str(row.get("ever_married", "")))
            self.Data_Table.setCellWidget(i, 8, married_cb)

            # Work Type
            work_cb = QComboBox()
            work_cb.addItems(["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            work_cb.setCurrentText(str(row.get("work_type", "")))
            self.Data_Table.setCellWidget(i, 9, work_cb)

            # Residence Type
            residence_cb = QComboBox()
            residence_cb.addItems(["Urban", "Rural"])
            residence_cb.setCurrentText(str(row.get("Residence_type", "")))
            self.Data_Table.setCellWidget(i, 10, residence_cb)

            # Glucose, BMI
            self.Data_Table.setItem(i, 11, QTableWidgetItem(str(row.get("avg_glucose_level", ""))))
            self.Data_Table.setItem(i, 12, QTableWidgetItem(str(row.get("bmi", ""))))

            # Smoking Status
            smoke_cb = QComboBox()
            smoke_cb.addItems(["never smoked", "formerly smoked", "smokes", "Unknown", "other"])
            smoke_cb.setCurrentText(str(row.get("smoking_status", "")))
            self.Data_Table.setCellWidget(i, 13, smoke_cb)

        self.Data_Table.resizeColumnsToContents()
        
    def add_blank_row(self):
        selected = self.Data_Table.currentRow()
        insert_index = self.Data_Table.rowCount() if selected == -1 else selected + 1

        self.Data_Table.insertRow(insert_index)

        # Insert row content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ID (temporary placeholder, will renumber all later)
        self.Data_Table.setItem(insert_index, 0, QTableWidgetItem(""))          # ID
        self.Data_Table.setItem(insert_index, 1, QTableWidgetItem(""))          # Name (blank)
        self.Data_Table.setItem(insert_index, 2, QTableWidgetItem(timestamp))   # Time

        # Result (blank + read-only)
        result_item = QTableWidgetItem("")
        result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
        self.Data_Table.setItem(insert_index, 3, result_item)

        # Age to BMI (blank)
        for col in [5, 6, 7, 11, 12]:
            self.Data_Table.setItem(insert_index, col, QTableWidgetItem(""))

        # ComboBoxes for categorical fields
        combo_fields = {
            4: ["Male", "Female", "Other"],
            8: ["Yes", "No"],
            9: ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            10: ["Urban", "Rural"],
            13: ["never smoked", "formerly smoked", "smokes", "Unknown", "other"]
        }

        for col_idx, options in combo_fields.items():
            combo = QComboBox()
            combo.addItems(options)
            self.Data_Table.setCellWidget(insert_index, col_idx, combo)

        # Renumber all IDs sequentially
        for row in range(self.Data_Table.rowCount()):
            self.Data_Table.setItem(row, 0, QTableWidgetItem(str(row + 1)))  # ID

        self.Data_Table.resizeColumnsToContents()
        
    def delete_row(self):
        row_to_delete = self.Data_Table.currentRow()
    
        if row_to_delete == -1:  # No selection
            row_to_delete = self.Data_Table.rowCount() - 1
    
        if row_to_delete >= 0:
            self.Data_Table.removeRow(row_to_delete)

    def analyze_data(self):
        # Ask user where to save
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        save_dir = os.path.join(root_dir, 'Dataset', 'User_Saved_SP')
        os.makedirs(save_dir, exist_ok=True)
        default_path = os.path.join(save_dir, 'user_saved.csv')

        file_path, _ = QFileDialog.getSaveFileName(self, "Save File Before Inference", default_path, "CSV Files (*.csv)")
        if not file_path:
            QMessageBox.warning(self, "Export Required", "You must export the table before running inference.")
            return

        #1. Pre-fill Result column with 0s (so stroke column won't be empty)
        for i in range(self.Data_Table.rowCount()):
            temp_item = QTableWidgetItem("Unlikely")  # Placeholder
            temp_item.setFlags(temp_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, temp_item)

        # === STEP 2: FORCE EXPORT (TO INCLUDE STROKE COLUMN) ===
        self.export_table_data(file_path)

        # === STEP 3: RUN INFERENCE ===
        model = SPModel(dataset_path=file_path)
        try:
            predictions = model.inference()
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"Failed to run inference:\n{str(e)}")
            return

        # === STEP 4: UPDATE Result COLUMN WITH FINAL OUTPUT ===
        for i, pred in enumerate(predictions):
            label = "Likely" if pred == 1 else "Unlikely"
            color = QColor(255, 100, 100) if pred == 1 else QColor(100, 255, 100)

            result_item = QTableWidgetItem(label)
            result_item.setBackground(color)
            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
            self.Data_Table.setItem(i, 3, result_item)

        self.Data_Table.resizeColumnsToContents()


from Eye import EyeWindow
from Skin import SkinWindow
from EMS import EMSWindow
