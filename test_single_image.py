import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = r"<ADD_YOUR_MODEL_PATH>/currency_model.h5"
IMG_PATH = r"<ADD_YOUR_IMAGE_PATH>/example.jpg"
IMG_SIZE = 224

class_indices = {
    0: '10 Rupees',
    1: '100 Rupees',
    2: '20 Rupees',
    3: '200 Rupees',
    4: '50 Rupees',
    5: '500 Rupees',
    6: 'Others'
}

model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(IMG_PATH):
    print(f"❌ Image not found: {IMG_PATH}")
else:
    # Load and preprocess image
    img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    pred = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(pred[0])
    pred_label = class_indices[pred_idx]

    print(f"\n✅ Predicted class → {pred_label}")


"""

Project: Handheld Currency Detection Device for the Visually Impaired

This script is part of a project to recognize Indian currency notes offline.
Refer to README.md for dataset structure, training instructions, and deployment on Raspberry Pi.

Team Members:
- 1JT21CS003 Achal N
- 1JT21CS019 Aruna A Shenoy
- 1JT21CS030 Bhushan M V
- 1JT21CS128 Pujitha D R

"""
