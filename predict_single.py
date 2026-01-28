from google.colab import drive
drive.mount('/content/drive')
from tensorflow.keras.models import load_model

model = load_model(
    "/content/drive/MyDrive/pcos_cnn_final_auc_0.998.h5",
    compile=False
)
print("Model loaded")
import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_single_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0              # SAME rescaling
    img = np.expand_dims(img, axis=0)  # shape (1,224,224,3)
    return img
def predict_single_image(img_path, threshold=0.5, min_confidence=0.75):
    img = preprocess_single_image(img_path)

    # Sigmoid output = P(noninfected)
    prob = model.predict(img, verbose=0)[0][0]

    if prob >= threshold:
        label = "noninfected"
        confidence = prob
    else:
        label = "infected"
        confidence = 1 - prob

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")

    # Confidence gate
    if confidence < min_confidence:
        print("⚠️ Warning: Low confidence prediction.")
        print("⚠️ Image may be outside the trained domain (e.g., non-medical image).")

    return label, confidence
import os

def predict_folder(folder_path, threshold=0.5, min_confidence=0.75):
    results = []

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            img = preprocess_single_image(img_path)

            # Sigmoid output = P(noninfected)
            prob = model.predict(img, verbose=0)[0][0]

            if prob >= threshold:
                label = "noninfected"
                confidence = prob
            else:
                label = "infected"
                confidence = 1 - prob

            # Confidence-based flag
            flag = "OK"
            if confidence < min_confidence:
                flag = "⚠️ Low confidence (possible non-medical image)"

            results.append((file, label, confidence, flag))

            print(
                f"{file:20s} → {label:12s} "
                f"({confidence*100:.2f}%)  {flag}"
            )

    return results

