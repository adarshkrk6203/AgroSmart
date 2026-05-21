import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json

# CONFIG
IMG_SIZE = 224
BATCH_SIZE = 32
test_dir = "dataset/test"

# Load model
model = tf.keras.models.load_model("disease_model.h5")

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_labels = list(class_indices.keys())

# Data generator (NO augmentation)
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predictions
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))