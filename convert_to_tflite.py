import tensorflow as tf

MODEL_H5_PATH = r"<ADD_YOUR_MODEL_PATH>/currency_model.h5"
MODEL_TFLITE_PATH = r"<ADD_YOUR_MODEL_PATH>/currency_model.tflite"

model = tf.keras.models.load_model(MODEL_H5_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Converted and saved TFLite model at {MODEL_TFLITE_PATH}")


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
