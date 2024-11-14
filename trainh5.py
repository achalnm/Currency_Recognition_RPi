import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

DATASET_DIR = r"<ADD_YOUR_DATASET_PATH>"
MODEL_H5_PATH = r"<ADD_MODEL_SAVE_PATH>/currency_model.h5"
MODEL_TFLITE_PATH = r"<ADD_MODEL_SAVE_PATH>/currency_model.tflite"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 75

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')

for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

num_classes = train_generator.num_classes

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(MODEL_H5_PATH, save_best_only=True)
earlystop_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print(f"✅ Model saved to {MODEL_H5_PATH}")
print("Class indices mapping:", train_generator.class_indices)

print("\n🔄 Converting model to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved to {MODEL_TFLITE_PATH}")


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
