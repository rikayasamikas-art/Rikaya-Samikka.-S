import json
import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Optional packages used only when DICOM/3D data is available
try:
    import pydicom
except Exception:
    pydicom = None

import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI-Based Brain Stroke Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "brain_stroke_model.keras"
PATIENT_RECORDS_DIR = PROJECT_DIR / "patient_records"
PATIENT_RECORDS_DIR.mkdir(exist_ok=True)

# IMPORTANT:
# This mapping is fixed according to the trained model's confirmed order.
CLASS_NAMES = {
    0: "Bleeding",
    1: "Ischemic",
    2: "Normal",
}

CLASS_INFO = {
    "Bleeding": "The model predicts a bleeding-type stroke pattern.",
    "Ischemic": "The model predicts an ischemic-type stroke pattern.",
    "Normal": "The model predicts no stroke pattern among the trained classes.",
}


# ============================================================
# PAGE STYLE
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 25px;
    }
    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }
    .result-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin: 10px 0;
    }
    .small-note {
        color: #666;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🧠 AI-Based Brain Stroke Detection</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">CT Scan Classification and 3D Visualization System</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def get_model():
    try:
        model = load_model()
        if model is None:
            st.error(
                "Model file not found. Please place "
                "`brain_stroke_model.keras` in the same GitHub repository as `app.py`."
            )
        return model
    except Exception as e:
        st.error(f"Unable to load the model: {e}")
        return None


# ============================================================
# PATIENT RECORD FUNCTIONS
# ============================================================
def safe_patient_filename(patient_id: str) -> str:
    cleaned = "".join(ch for ch in patient_id if ch.isalnum() or ch in ("-", "_"))
    return cleaned[:100] or "unknown_patient"


def patient_record_path(patient_id: str) -> Path:
    return PATIENT_RECORDS_DIR / f"{safe_patient_filename(patient_id)}.json"


def load_patient_record(patient_id: str):
    path = patient_record_path(patient_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_patient_record(record: dict):
    path = patient_record_path(record["patient_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


# ============================================================
# IMAGE / PREDICTION FUNCTIONS
# ============================================================
def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()

    image = image.resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return display_image, array


def predict_image(model, image_array):
    raw = model.predict(image_array, verbose=0)
    probabilities = np.asarray(raw).reshape(-1)

    if probabilities.size != 3:
        raise ValueError(
            f"Expected model output with 3 classes, but received shape {raw.shape}."
        )

    # Softmax is applied only if the output does not already look like probabilities.
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=0.05):
        exp_values = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_values / exp_values.sum()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    probability_dict = {
        CLASS_NAMES[i]: float(probabilities[i] * 100)
        for i in range(3)
    }

    return predicted_class, confidence, probability_dict


# ============================================================
# PDF REPORT
# ============================================================
def create_pdf(record: dict) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 55

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(55, y, "AI-Based Brain Stroke Detection Report")

    y -= 35
    pdf.setFont("Helvetica", 11)

    fields = [
        ("Patient Name", record.get("patient_name", "")),
        ("Patient ID", record.get("patient_id", "")),
        ("Gender", record.get("gender", "")),
        ("Address", record.get("address", "")),
        ("Scan Date", record.get("scan_date", "")),
        ("Predicted Class", record.get("predicted_class", "")),
        ("Confidence", f'{record.get("confidence", 0):.2f}%'),
    ]

    for label, value in fields:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(55, y, f"{label}:")
        pdf.setFont("Helvetica", 11)

        text = str(value)
        if len(text) > 75:
            text = text[:72] + "..."
        pdf.drawString(175, y, text)
        y -= 22

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(55, y, "Class Probabilities")
    y -= 22

    pdf.setFont("Helvetica", 11)
    probabilities = record.get("probabilities", {})
    for class_name in ("Bleeding", "Ischemic", "Normal"):
        pdf.drawString(
            75,
            y,
            f"{class_name}: {float(probabilities.get(class_name, 0)):.2f}%",
        )
        y -= 20

    y -= 15
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(55, y, "Interpretation")
    y -= 20

    pdf.setFont("Helvetica", 10)
    interpretation = CLASS_INFO.get(
        record.get("predicted_class", ""),
        "Prediction generated by the trained machine-learning model.",
    )

    # Simple line wrapping
    words = interpretation.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 90:
            pdf.drawString(55, y, line)
            y -= 16
            line = word
        else:
            line = (line + " " + word).strip()

    if line:
        pdf.drawString(55, y, line)
        y -= 20

    y -= 20
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        55,
        y,
        "This report is generated for academic/project demonstration purposes.",
    )

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# DICOM + MASK HELPERS
# ============================================================
def load_uploaded_dicom_zip(uploaded_zip):
    """
    Reads a ZIP containing DICOM files and optional mask files.

    Supported arrangement:
      - .dcm files
      - mask.npy / mask.npz files
      - image.npy / image.npz files

    If a mask is not supplied with the selected study, the 3D affected
    region is reported as unavailable rather than reusing another patient.
    """
    if uploaded_zip is None:
        return None, None, None

    try:
        data = uploaded_zip.getvalue()
        zf = zipfile.ZipFile(io.BytesIO(data))

        dcm_names = [
            n for n in zf.namelist()
            if n.lower().endswith(".dcm") and not n.endswith("/")
        ]

        if not dcm_names:
            return None, None, "No DICOM files were found in the ZIP."

        slices = []
        for name in dcm_names:
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
                arr = ds.pixel_array.astype(np.float32)

                # Basic sorting using InstanceNumber when available.
                instance = getattr(ds, "InstanceNumber", 0)
                slices.append((int(instance), name, arr))
            except Exception:
                continue

        if not slices:
            return None, None, "DICOM files could not be read."

        slices.sort(key=lambda x: (x[0], x[1]))
        volume = np.stack([x[2] for x in slices], axis=0)

        mask_volume = None

        mask_candidates = [
            n for n in zf.namelist()
            if (
                n.lower().endswith(".npy")
                or n.lower().endswith(".npz")
            )
            and "mask" in n.lower()
        ]

        if mask_candidates:
            mask_name = mask_candidates[0]
            raw = zf.read(mask_name)
            loaded = np.load(io.BytesIO(raw), allow_pickle=False)

            if isinstance(loaded, np.lib.npyio.NpzFile):
                keys = list(loaded.keys())
                if keys:
                    mask_volume = loaded[keys[0]]
            else:
                mask_volume = loaded

            mask_volume = np.asarray(mask_volume)

            if mask_volume.ndim == 2:
                mask_volume = np.expand_dims(mask_volume, axis=0)

            if mask_volume.shape != volume.shape:
                mask_volume = None

        return volume, mask_volume, None

    except Exception as e:
        return None, None, f"Unable to process DICOM ZIP: {e}"


def make_3d_figure(volume, mask=None):
    # Downsample for browser performance.
    max_dim = 96
    step_z = max(1, int(np.ceil(volume.shape[0] / max_dim)))
    step_y = max(1, int(np.ceil(volume.shape[1] / max_dim)))
    step_x = max(1, int(np.ceil(volume.shape[2] / max_dim)))

    vol = volume[::step_z, ::step_y, ::step_x]

    # Normalize CT values for visualization.
    vmin = np.percentile(vol, 2)
    vmax = np.percentile(vol, 98)
    normalized = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)

    z, y, x = np.where(normalized > 0.35)

    # Keep point count manageable.
    if len(x) > 18000:
        idx = np.linspace(0, len(x) - 1, 18000).astype(int)
        x, y, z = x[idx], y[idx], z[idx]
        values = normalized[z, y, x]
    else:
        values = normalized[z, y, x]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=2,
                opacity=0.18,
                color=values,
                colorscale="Gray",
                showscale=False,
            ),
            name="CT Volume",
        )
    )

    if mask is not None and mask.shape == volume.shape:
        m = mask[::step_z, ::step_y, ::step_x]
        mz, my, mx = np.where(m > 0)

        if len(mx) > 25000:
            idx = np.linspace(0, len(mx) - 1, 25000).astype(int)
            mx, my, mz = mx[idx], my[idx], mz[idx]

        if len(mx) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=mx,
                    y=my,
                    z=mz,
                    mode="markers",
                    marker=dict(
                        size=3,
                        opacity=0.65,
                        color="red",
                    ),
                    name="Affected Region",
                )
            )

    fig.update_layout(
        title="3D CT Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# ============================================================
# SECTION 1 — PATIENT INFORMATION
# ============================================================
st.markdown(
    '<div class="section-title">1. Patient Information</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

with col2:
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
    scan_date = st.date_input("Scan Date", value=date.today())

address = st.text_area("Address")

check_patient = st.button("Check Patient ID", type="secondary")


# ============================================================
# EXISTING PATIENT LOOKUP
# ============================================================
existing_record = None

if check_patient:
    if not patient_id.strip():
        st.warning("Please enter a Patient ID.")
    else:
        existing_record = load_patient_record(patient_id.strip())

        if existing_record:
            st.success("Existing patient record found.")

            st.subheader("Saved Patient Report")

            saved_col1, saved_col2 = st.columns(2)

            with saved_col1:
                st.write(f"**Patient Name:** {existing_record.get('patient_name', '')}")
                st.write(f"**Patient ID:** {existing_record.get('patient_id', '')}")
                st.write(f"**Gender:** {existing_record.get('gender', '')}")
                st.write(f"**Address:** {existing_record.get('address', '')}")

            with saved_col2:
                st.write(f"**Scan Date:** {existing_record.get('scan_date', '')}")
                st.write(
                    f"**Prediction:** {existing_record.get('predicted_class', '')}"
                )
                st.write(
                    f"**Confidence:** {float(existing_record.get('confidence', 0)):.2f}%"
                )

            probs = existing_record.get("probabilities", {})
            st.write("**Class probabilities:**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Class": ["Bleeding", "Ischemic", "Normal"],
                        "Probability (%)": [
                            float(probs.get("Bleeding", 0)),
                            float(probs.get("Ischemic", 0)),
                            float(probs.get("Normal", 0)),
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            pdf_bytes = create_pdf(existing_record)
            st.download_button(
                "Download Saved PDF Report",
                data=pdf_bytes,
                file_name=f"{safe_patient_filename(patient_id)}_report.pdf",
                mime="application/pdf",
            )

        else:
            st.info(
                "No saved record was found for this Patient ID. "
                "Continue with a new CT scan."
            )


# ============================================================
# SECTION 2 — NEW PATIENT CT PREDICTION
# ============================================================
st.markdown(
    '<div class="section-title">2. New CT Scan Prediction</div>',
    unsafe_allow_html=True,
)

st.info(
    "For a new patient, upload a CT image. The model predicts "
    "Bleeding, Ischemic, or Normal."
)

uploaded_image = st.file_uploader(
    "Upload CT Scan Image",
    type=["png", "jpg", "jpeg"],
    key="ct_image",
)

# Optional DICOM ZIP for 3D visualization.
uploaded_dicom = st.file_uploader(
    "Optional: Upload matching DICOM + mask ZIP for 3D visualization",
    type=["zip"],
    key="dicom_zip",
)

predict_button = st.button("Predict Stroke Type", type="primary")


if predict_button:
    # Validate patient fields.
    if not patient_name.strip():
        st.warning("Please enter Patient Name.")
        st.stop()

    if not patient_id.strip():
        st.warning("Please enter Patient ID.")
        st.stop()

    if gender == "Select":
        st.warning("Please select Gender.")
        st.stop()

    if not address.strip():
        st.warning("Please enter Address.")
        st.stop()

    if uploaded_image is None:
        st.warning("Please upload a CT image.")
        st.stop()

    # Prevent accidental overwriting of an existing patient report.
    old_record = load_patient_record(patient_id.strip())
    if old_record is not None:
        st.warning(
            "This Patient ID already has a saved report. "
            "Use 'Check Patient ID' to view the old report instead of "
            "creating another report with the same ID."
        )
        st.stop()

    model = get_model()
    if model is None:
        st.stop()

    try:
        display_image, input_array = prepare_image(uploaded_image)
        predicted_class, confidence, probabilities = predict_image(
            model, input_array
        )

        st.session_state["latest_prediction"] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

        record = {
            "patient_name": patient_name.strip(),
            "patient_id": patient_id.strip(),
            "gender": gender,
            "address": address.strip(),
            "scan_date": str(scan_date),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

        save_patient_record(record)

        st.success("Prediction completed and patient report saved.")

        left, right = st.columns(2)

        with left:
            st.image(display_image, caption="Uploaded CT Scan", use_container_width=True)

        with right:
            st.subheader("Prediction Result")
            st.metric("Predicted Class", predicted_class)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.write("### Class Probabilities")
            probability_df = pd.DataFrame(
                {
                    "Class": ["Bleeding", "Ischemic", "Normal"],
                    "Probability (%)": [
                        probabilities["Bleeding"],
                        probabilities["Ischemic"],
                        probabilities["Normal"],
                    ],
                }
            )
            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True,
            )

        st.markdown(
            f'<div class="result-box"><b>Interpretation:</b> '
            f'{CLASS_INFO[predicted_class]}</div>',
            unsafe_allow_html=True,
        )

        # ========================================================
        # SECTION 3 — 3D VISUALIZATION
        # ========================================================
        st.markdown(
            '<div class="section-title">3. 3D Visualization</div>',
            unsafe_allow_html=True,
        )

        if uploaded_dicom is None:
            st.info(
                "3D affected-region visualization is Not available because "
                "a matching DICOM + mask dataset was not provided for this CT."
            )
        elif pydicom is None:
            st.error(
                "pydicom is not installed. Add pydicom to requirements.txt."
            )
        else:
            volume, mask_volume, error = load_uploaded_dicom_zip(uploaded_dicom)

            if error:
                st.warning(error)
            elif volume is None:
                st.info("3D visualization is Not available.")
            else:
                st.write(
                    f"CT volume loaded successfully: {volume.shape}"
                )

                if mask_volume is None:
                    st.info(
                        "Matching mask was not found for this DICOM study. "
                        "Affected region: Not available."
                    )
                else:
                    affected_voxels = int(np.count_nonzero(mask_volume > 0))

                    if affected_voxels == 0:
                        st.info(
                            "The matching mask contains no affected region. "
                            "Affected region: Not available."
                        )
                    else:
                        st.success(
                            f"Matching mask found. Affected voxels: {affected_voxels:,}"
                        )

                        fig = make_3d_figure(volume, mask_volume)
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ============================================================
# SECTION 4 — PDF DOWNLOAD
# ============================================================
latest = st.session_state.get("latest_prediction")

if latest is not None:
    st.markdown(
        '<div class="section-title">4. PDF Report</div>',
        unsafe_allow_html=True,
    )

    pdf_record = {
        "patient_name": patient_name.strip(),
        "patient_id": patient_id.strip(),
        "gender": gender,
        "address": address.strip(),
        "scan_date": str(scan_date),
        "predicted_class": latest["predicted_class"],
        "confidence": latest["confidence"],
        "probabilities": latest["probabilities"],
    }

    pdf_bytes = create_pdf(pdf_record)

    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name=f"{safe_patient_filename(patient_id)}_brain_stroke_report.pdf",
        mime="application/pdf",
        type="primary",
    )

st.markdown("---")
st.caption(
    "Academic project prototype — AI prediction should not be treated as a clinical diagnosis."
)
