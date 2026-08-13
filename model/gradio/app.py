import gradio as gr
import numpy as np
import cv2
import joblib
import os

# ================= LOAD MODEL =================
MODEL_PATH = "random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Train and save Random Forest first.")

model = joblib.load(MODEL_PATH)

# ================= CATEGORY LABELS =================
# ⚠️ IMPORTANT: Replace with your actual folder names in SAME ORDER
categories = ["BLOT", "histopathology", "x-ray"]


# ================= PREDICTION FUNCTION =================
def predict_image(image):
    try:
        # Convert to numpy
        img = np.array(image)

        # Resize
        img = cv2.resize(img, (64, 64))

        # Normalize
        img = img / 255.0

        # Flatten
        img = img.flatten().reshape(1, -1)

        # Predict
        pred = model.predict(img)[0]
        label = categories[pred]

        return f"Predicted Class: {label}"

    except Exception as e:
        return f"Error: {str(e)}"


# ================= GRADIO UI =================
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Scientific Image Classification",
    description="Upload an image to predict its category using Random Forest"
)

interface.launch()