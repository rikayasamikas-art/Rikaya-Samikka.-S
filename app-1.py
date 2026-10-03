import os
import io
import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
import pydicom
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/BrainStroke_Project"
DATASET_DIR = "/content/drive/MyDrive/dataset"
MODEL_PATH = os.path.join(PROJECT_DIR, "brain_stroke_model.keras")
PATIENT_RECORDS_DIR = os.path.join(PROJECT_DIR, "patient_records")

CLASS_NAMES = {
    0: "Bleeding",
    1: "Ischemic",
    2: "Normal",
}


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Brain Stroke Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Brain Stroke Detection")
st.caption("CT Prediction + DICOM 3D Visualization")


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    arr = np.array(image, dt
