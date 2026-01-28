from google.colab import drive
drive.mount('/content/drive')
"/content/drive/MyDrive/PCOS"
import os

base_dir = "/content/drive/MyDrive/PCOS"
print(os.listdir(base_dir))
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
base_dir = "/content/drive/MyDrive/PCOS"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,  # 70% train, 15% val, 15% test (approx)
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)
val_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12
)
val_generator.reset()
preds = model.predict(val_generator)

y_pred = (preds >= 0.5).astype(int).flatten()
y_true = val_generator.classes

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_true, y_pred))
print(classification_report(
    y_true,
    y_pred,
    target_names=list(val_generator.class_indices.keys())
))
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return -tf.reduce_mean(
            alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) +
            (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        )
    return loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)
class_weight = {
    0: 1.0,   # infected
    1: 3.0    # noninfected
}
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,
    class_weight=class_weight
)
val_generator.reset()
preds = model.predict(val_generator)

y_pred = (preds >= 0.5).astype(int).flatten()
y_true = val_generator.classes

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_true, y_pred))
print(classification_report(
    y_true,
    y_pred,
    target_names=list(val_generator.class_indices.keys())
))
print("Train samples:", train_generator.samples)
print("Val samples:", val_generator.samples)

print(
    "Overlap:",
    len(set(train_generator.filenames).intersection(
        set(val_generator.filenames)
    ))
)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(val_generator.classes, preds)
print("ROC-AUC:", auc)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(val_generator.classes, preds)
plt.plot(fpr, tpr, label="AUC = 0.998")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

