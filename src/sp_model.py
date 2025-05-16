import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#import onnxruntime as ort
from rknnlite.api import RKNNLite

 


class SPModel:
    #def __init__(self, dataset_path=None, model_path=None):
    #    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    #    self.dataset_path = dataset_path or os.path.join(root_dir, 'Dataset', 'SP_sample.csv')
    #    self.model_path = model_path or os.path.join(root_dir, 'Models', 'SP_91.onnx')
    #    self.session = ort.InferenceSession(self.model_path)
    #    self.input_name = self.session.get_inputs()[0].name
    
    def __init__(self, dataset_path=None, model_path=None):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.dataset_path = dataset_path or os.path.join(root_dir, 'Dataset', 'SP_sample.csv')
        self.model_path = model_path or os.path.join(root_dir, 'Models', 'SP_91.rknn')
        self.batch_size = 4
        
        # Initialize RKNNLite
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model")

        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")
    

    def preprocess_sample_data(self):
        df = pd.read_csv(self.dataset_path)
        df = df.drop(columns=["id", "name", "time_modified", "result"], errors='ignore')

        df["age_group"] = df["age"].apply(lambda x: "Infant" if (x >= 0) & (x <= 2)
            else ("Child" if (x > 2) & (x <= 12)
            else ("Adolescent" if (x > 12) & (x <= 18)
            else ("Young Adults" if (x > 19) & (x <= 35)
            else ("Middle Aged Adults" if (x > 35) & (x <= 60)
            else "Old Aged Adults")))))

        df['bmi'] = df['bmi'].fillna(df.groupby(["gender", "ever_married", "age_group"])["bmi"].transform('mean'))
        df = df[(df["bmi"] < 66) & (df["bmi"] > 12)]
        df = df[(df["avg_glucose_level"] > 56) & (df["avg_glucose_level"] < 250)]
        df = df[df["gender"].isin(["Male", "Female"])]

        #had_stroke = df[df["stroke"] == 1]
        #no_stroke = df[df["stroke"] == 0]
        #upsampled_had_stroke = resample(had_stroke, replace=True, n_samples=no_stroke.shape[0], random_state=123)
        #upsampled_data = pd.concat([no_stroke, upsampled_had_stroke])

        categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        dummies = pd.get_dummies(df[categorical], dtype=int)

        expected_dummy_cols = [
            'gender_Female', 'gender_Male',
            'ever_married_No', 'ever_married_Yes',
            'work_type_Govt_job', 'work_type_Never_worked',
            'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Rural', 'Residence_type_Urban',
            'smoking_status_Unknown', 'smoking_status_formerly smoked',
            'smoking_status_never smoked', 'smoking_status_smokes'
        ]
        for col in expected_dummy_cols:
            if col not in dummies:
                dummies[col] = 0
        dummies = dummies[expected_dummy_cols]

        model_data = pd.concat([df.drop(columns=categorical), dummies], axis=1)

        encoder = LabelEncoder()
        model_data["age_group"] = encoder.fit_transform(model_data["age_group"])

        scaler = MinMaxScaler()
        for col in ['age', 'avg_glucose_level', 'bmi']:
            scaler.fit(model_data[[col]])
            model_data[col] = scaler.transform(model_data[[col]])

        X_input = model_data.drop(columns=["stroke"]).astype(np.float32).values
        y_true = model_data["stroke"].astype(int).tolist()
        return X_input, y_true

    #def inference(self):
    #    X_input, _ = self.preprocess_sample_data()
    #    results = []
#
    #    for i in range(len(X_input)):
    #        input_data = np.expand_dims(X_input[i], axis=0)
    #        output = self.session.run(None, {self.input_name: input_data})
    #        prediction = int(output[0][0][0] > 0.5)
    #        results.append(prediction)
#
    #    return results
    
    def inference(self):
        X_input, _ = self.preprocess_sample_data()
        results = []

        # Pad input to match batch size
        pad_len = (-len(X_input)) % self.batch_size
        if pad_len > 0:
            pad = np.zeros((pad_len, X_input.shape[1]), dtype=np.float32)
            X_input = np.concatenate([X_input, pad], axis=0)

        for i in range(0, len(X_input), self.batch_size):
            batch = X_input[i:i + self.batch_size]
            output = self.rknn.inference(inputs=[batch])
            if output is not None:
                preds = (output[0].squeeze() > 0.5).astype(int).tolist()
                results.extend(preds)
            else:
                results.extend([0] * self.batch_size)

        # Remove padding
        if pad_len > 0:
            results = results[:-pad_len]

        return results
    

#How to Use
#model = SPModel()
#predictions = model.inference()
#print(predictions)