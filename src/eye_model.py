import os
import numpy as np
import pandas as pd
from PIL import Image
#import onnxruntime as ort
from rknnlite.api import RKNNLite


class EYEModel:
    #def __init__(self, sample_dir=None, model_path=None):
    #    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    #    self.sample_dir = sample_dir or os.path.join(root_dir, "Dataset", "EYE_sample")
    #    self.model_path = model_path or os.path.join(root_dir, "Models", "EYE_81.onnx")
    #    
    #    self.session = ort.InferenceSession(self.model_path)
    #    self.input_name = self.session.get_inputs()[0].name
    
    def __init__(self, sample_dir=None, model_path=None):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.sample_dir = sample_dir or os.path.join(root_dir, 'Dataset', 'EYE_sample')
        self.model_path = model_path or os.path.join(root_dir, 'Models', 'EYE_81.rknn')
        self.batch_size = 3

        self.class_names = [
            "cataract",
            "diabetic_retinopathy",
            "glaucoma",
            "normal"
        ]

        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(self.model_path) != 0:
            raise RuntimeError("Failed to load RKNN model")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
            raise RuntimeError("Failed to init RKNN runtime")

    def preprocess_sample_data(self, image_size=224):
        X = []
        for fname in os.listdir(self.sample_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(self.sample_dir, fname)
                img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
                img_arr = np.array(img, dtype=np.float32) / 255.0
                X.append(img_arr)
        X = np.stack(X, axis=0)
        return X
    
    #def inference(self):
    #    class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    #    X = self.preprocess_sample_data()
    #    results = []
    #    for i in range(len(X)):
    #        img = np.expand_dims(X[i], axis=0)
    #        output = self.session.run(None, {self.input_name: img})
    #        pred_idx = int(np.argmax(output[0], axis=1)[0])
    #        results.append(class_names[pred_idx])
    #    return results
    
    def inference(self):
        X_img = self.preprocess_sample_data()
        results = []

        pad_len = (-len(X_img)) % self.batch_size
        if pad_len > 0:
            pad = np.zeros((pad_len, 224, 224, 3), dtype=np.float32)
            X_img = np.concatenate([X_img, pad], axis=0)

        for i in range(0, len(X_img), self.batch_size):
            batch = X_img[i:i + self.batch_size]
            output = self.rknn.inference(inputs=[batch], data_format='nchw')
            if output is not None:
                preds = np.argmax(output[0], axis=1)
                results.extend([self.class_names[idx] for idx in preds])
            else:
                results.extend(["unknown"] * self.batch_size)

        if pad_len > 0:
            results = results[:-pad_len]

        return results

# How to use
#model = EYEModel()
#predictions = model.inference()
#print(predictions)
