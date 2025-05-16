import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import onnxruntime as ort


class SKINModel:
    def __init__(self, sample_dir=None, model_path=None):
        # Resolve paths relative to project root
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.sample_dir = sample_dir
        self.model_path = model_path or os.path.join(root_dir, "Models", "SKIN_69.onnx")
        
        self.session = ort.InferenceSession(self.model_path)
        self.img_input_name = self.session.get_inputs()[0].name
        self.meta_input_name = self.session.get_inputs()[1].name

    def preprocess_sample_data(self, image_size=224):
        
        # Dynamically find the first .csv file in sample_dir
        csv_path = None
        for file in os.listdir(self.sample_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(self.sample_dir, file)
                break

        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError("No valid CSV file found in the sample directory.")
        
        df = pd.read_csv(csv_path)
        df.drop(columns=["result", "id", "name", "time_modified", "Take Photo"], errors="ignore", inplace=True) #For custom dataset
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(self.sample_dir, f"{x}.jpg"))

        # Load and normalize images
        X_img = []
        for img_id in df['image_id']:
            img = Image.open(os.path.join(self.sample_dir, f"{img_id}.jpg"))\
                .convert("RGB").resize((image_size, image_size))
            X_img.append(np.array(img, np.float32) / 255.0)
        X_img = np.stack(X_img, axis=0)

        # Tabular (meta) data
        meta_df = df[['age', 'sex', 'localization']].copy()
        meta_df['age'].fillna(meta_df['age'].mean(), inplace=True)

        # Define category lists
        sex_categories = ['male', 'female']
        loc_categories = [
            'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
            'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity'
        ]

        # Fixed one-hot encoder (no refitting)
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['age']),
            ('cat', OneHotEncoder(categories=[sex_categories, loc_categories], handle_unknown='ignore'),
             ['sex', 'localization'])
        ])

        X_meta = preprocessor.fit_transform(meta_df).astype(np.float32)
        if hasattr(X_meta, "toarray"):
            X_meta = X_meta.toarray()
        X_meta = np.asarray(X_meta, dtype=np.float32)

        return X_img, X_meta

    def inference(self):
        class_names = [
            'Actinic Keratoses',
            'Basal Cell Carcinoma',
            'Benign Keratosis-like Lesions',
            'Dermatofibroma',
            'Melanocytic Nevi',
            'Melanoma',
            'Vascular Lesions'
        ]

        name_to_abbr = {
            'Actinic Keratoses': 'akiec',
            'Basal Cell Carcinoma': 'bcc',
            'Benign Keratosis-like Lesions': 'bkl',
            'Dermatofibroma': 'df',
            'Melanocytic Nevi': 'nv',
            'Melanoma': 'mel',
            'Vascular Lesions': 'vasc'
        }

        X_img, X_meta = self.preprocess_sample_data()
        results = []

        for i in range(len(X_img)):
            img = np.expand_dims(X_img[i], axis=0)
            meta = np.expand_dims(X_meta[i], axis=0)
            output = self.session.run(None, {
                self.img_input_name: img,
                self.meta_input_name: meta
            })

            pred_idx = int(np.argmax(output[0], axis=1)[0])
            full_name = class_names[pred_idx]
            abbr = name_to_abbr.get(full_name, 'unknown')
            results.append(abbr)

        return results

#How to Use
#model = SKINModel()
#predictions = model.inference()
#print(predictions)
