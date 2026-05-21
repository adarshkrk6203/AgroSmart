import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import json

# ================= CONFIG ================= #
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

train_dir = "dataset/train"
val_dir = "dataset/test"

# ================= DATA GENERATORS ================= #
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ================= SAVE CLASS LABELS ================= #
class_indices = train_data.class_indices

with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("Class indices saved.")

# ================= LOAD BASE MODEL ================= #
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# ================= BUILD MODEL ================= #
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================= CALLBACKS ================= #
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ================= TRAIN (PHASE 1) ================= #
print("Starting initial training...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ================= FINE-TUNING ================= #
print("Starting fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# ================= EVALUATION ================= #
print("Evaluating model...")

val_loss, val_acc = model.evaluate(val_data)

print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# ================= SAVE MODEL ================= #
model.save("disease_model.h5")

print("Model trained and saved as disease_model.h5")
print("Ready for Flask integration.")