
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI-Based Brain Stroke Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Brain Stroke Detection")
st.subheader("CT Scan Analysis")

MODEL_PATH = "/content/drive/MyDrive/BrainStroke_Project/brain_stroke_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ["Bleeding", "Ischemic", "Normal"]

uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded CT Scan",
        width=400
    )

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_index]) * 100

    predicted_class = class_names[predicted_index]

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")

    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i] * 100:.2f}%")
