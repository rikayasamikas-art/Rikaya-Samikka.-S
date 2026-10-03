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


def prepare_dicom_slice_for_model(volume):
    """
    Convert a DICOM CT volume into one representative 2D CT image for
    the existing 224x224 CNN.

    The mask is NOT used for choosing the slice, so the prediction does
    not depend on the ground-truth mask.
    """
    vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim != 3 or vol.shape[0] == 0:
        raise ValueError("Invalid DICOM volume.")

    # Pick a useful anatomical slice rather than blindly using slice 0.
    # Prefer slices with a substantial non-air area, then choose the
    # slice closest to the middle of that valid range.
    body_scores = []
    for i in range(vol.shape[0]):
        slice_i = vol[i]
        valid = np.count_nonzero(slice_i > -500)
        body_scores.append(valid)

    body_scores = np.asarray(body_scores)
    threshold = max(1000, int(0.05 * vol.shape[1] * vol.shape[2]))
    valid_indices = np.where(body_scores >= threshold)[0]

    if len(valid_indices) == 0:
        slice_index = vol.shape[0] // 2
    else:
        slice_index = int(valid_indices[len(valid_indices) // 2])

    ct_slice = vol[slice_index]

    # Robust CT windowing for display/model input.
    # This avoids feeding raw HU values directly into a model trained
    # on normalized images.
    lower, upper = -100.0, 200.0
    windowed = np.clip(ct_slice, lower, upper)
    windowed = (windowed - lower) / (upper - lower + 1e-8)
    image_u8 = (windowed * 255.0).astype(np.uint8)

    display_image = Image.fromarray(image_u8).convert("RGB")
    resized = display_image.resize((224, 224))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return display_image, array, slice_index


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
def _read_npz_array(raw_bytes):
    """Read a numpy/npz mask from ZIP bytes."""
    loaded = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.keys())
        if not keys:
            return None
        arr = loaded[keys[0]]
        loaded.close()
        return np.asarray(arr)
    return np.asarray(loaded)


def load_uploaded_dicom_zip(uploaded_zip):
    """
    Load a DICOM study ZIP and pair each raw.dcm slice with the mask.npz
    located beside that same slice.

    Supported layouts:
      1) study/00041/raw.dcm + study/00041/mask.npz
      2) folder/raw_001.dcm + folder/mask.npz (single volume mask)
      3) flat DICOM files + one mask.npy/mask.npz volume
    """
    if uploaded_zip is None:
        return None, None, None

    try:
        if pydicom is None:
            return None, None, "pydicom is not installed."

        data = uploaded_zip.getvalue()
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in zf.namelist() if not n.endswith("/")]

        # Prefer actual raw.dcm files. If not present, accept any .dcm.
        dcm_names = [n for n in names if n.lower().endswith("/raw.dcm")]
        if not dcm_names:
            dcm_names = [n for n in names if n.lower().endswith(".dcm")]

        if not dcm_names:
            return None, None, "No DICOM files were found in the ZIP."

        def slice_sort_key(name):
            parts = Path(name).parts
            # Dataset slices are usually .../00041/raw.dcm
            for part in reversed(parts[:-1]):
                if part.isdigit():
                    return (0, int(part), name)
            try:
                ds = pydicom.dcmread(
                    io.BytesIO(zf.read(name)),
                    stop_before_pixels=True,
                    force=True,
                )
                inst = int(getattr(ds, "InstanceNumber", 0))
            except Exception:
                inst = 0
            return (1, inst, name)

        dcm_names = sorted(dcm_names, key=slice_sort_key)

        slices = []
        paired_masks = []
        mask_found_for_slice = 0

        for name in dcm_names:
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
                arr = ds.pixel_array.astype(np.float32)

                # Convert stored pixel values to HU when metadata is available.
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

                slices.append(arr)

                # Match mask from the SAME slice directory.
                parent = str(Path(name).parent)
                sibling_candidates = [
                    f"{parent}/mask.npz",
                    f"{parent}/mask.npy",
                ]
                mask_arr = None
                for candidate in sibling_candidates:
                    if candidate in names:
                        mask_arr = _read_npz_array(zf.read(candidate))
                        break

                paired_masks.append(mask_arr)
                if mask_arr is not None:
                    mask_found_for_slice += 1

            except Exception:
                continue

        if not slices:
            return None, None, "DICOM files could not be read."

        volume = np.stack(slices, axis=0).astype(np.float32)

        # Case A: one mask beside every DICOM slice.
        if mask_found_for_slice == len(slices) and all(
            m is not None for m in paired_masks
        ):
            try:
                mask_volume = np.stack(
                    [np.asarray(m).squeeze() for m in paired_masks],
                    axis=0,
                )
                if mask_volume.shape != volume.shape:
                    mask_volume = None
            except Exception:
                mask_volume = None
        else:
            mask_volume = None

        # Case B: a single whole-volume mask exists somewhere in the ZIP.
        if mask_volume is None:
            volume_mask_candidates = [
                n for n in names
                if (
                    ("mask" in n.lower())
                    and (
                        n.lower().endswith(".npz")
                        or n.lower().endswith(".npy")
                    )
                )
                and not n.lower().endswith("/mask.npz")
                and not n.lower().endswith("/mask.npy")
            ]

            for mask_name in volume_mask_candidates:
                try:
                    candidate = _read_npz_array(zf.read(mask_name)).squeeze()
                    if candidate.shape == volume.shape:
                        mask_volume = candidate
                        break
                except Exception:
                    continue

        if mask_volume is not None:
            mask_volume = np.asarray(mask_volume)
            if mask_volume.ndim != 3 or mask_volume.shape != volume.shape:
                mask_volume = None

        if mask_found_for_slice == len(slices) and mask_volume is not None:
            mask_status = f"Matched {mask_found_for_slice}/{len(slices)} DICOM slice masks."
        elif mask_volume is not None:
            mask_status = "A matching whole-volume mask was found."
        elif mask_found_for_slice == 0:
            mask_status = "No matching mask was found in this DICOM ZIP."
        else:
            mask_status = (
                f"Only {mask_found_for_slice}/{len(slices)} slice masks were found; "
                "affected-region display is marked unavailable."
            )

        return volume, mask_volume, None, mask_status

    except Exception as e:
        return None, None, f"Unable to process DICOM ZIP: {e}", None


def make_3d_figure(volume, mask=None):
    """
    Browser-friendly 3D CT surface using Plotly isosurfaces.
    The CT is rendered as a soft-tissue surface instead of a raw point cloud.
    A matching mask is rendered as a separate affected-region surface.
    """
    vol = np.asarray(volume, dtype=np.float32)

    # Downsample while keeping the volume shape proportional.
    target = 64
    step_z = max(1, int(np.ceil(vol.shape[0] / target)))
    step_y = max(1, int(np.ceil(vol.shape[1] / target)))
    step_x = max(1, int(np.ceil(vol.shape[2] / target)))

    vol = vol[::step_z, ::step_y, ::step_x]

    # Soft-tissue CT window.
    lower, upper = -20.0, 120.0
    normalized = np.clip((vol - lower) / (upper - lower), 0, 1)

    z, y, x = np.indices(normalized.shape)

    fig = go.Figure()

    # Isosurface threshold suppresses most air and gives a continuous
    # anatomical surface rather than a cylinder-like marker cloud.
    fig.add_trace(
        go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=normalized.flatten(),
            isomin=0.30,
            isomax=0.95,
            surface_count=2,
            opacity=0.28,
            colorscale="Gray",
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="CT Brain Surface",
        )
    )

    if mask is not None and mask.shape == volume.shape:
        m = np.asarray(mask)[::step_z, ::step_y, ::step_x]
        mz, my, mx = np.indices(m.shape)

        affected = (m > 0).astype(np.float32)

        if np.any(affected > 0):
            fig.add_trace(
                go.Isosurface(
                    x=mx.flatten(),
                    y=my.flatten(),
                    z=mz.flatten(),
                    value=affected.flatten(),
                    isomin=0.5,
                    isomax=1.0,
                    surface_count=1,
                    opacity=0.80,
                    colorscale=[[0, "red"], [1, "red"]],
                    showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    name="Affected Region",
                )
            )

    fig.update_layout(
        title="3D CT Brain Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="white",
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h"),
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
# SECTION 2 — NEW PATIENT CT / DICOM PREDICTION
# ============================================================
st.markdown(
    '<div class="section-title">2. New CT Scan Prediction</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload either a CT image (PNG/JPG) or a DICOM study ZIP. "
    "For the dataset format, each slice can contain raw.dcm + matching "
    "mask.npz; the app pairs masks with their own DICOM slices."
)

uploaded_image = st.file_uploader(
    "Upload CT Scan Image (optional when DICOM ZIP is provided)",
    type=["png", "jpg", "jpeg"],
    key="ct_image",
)

uploaded_dicom = st.file_uploader(
    "Upload DICOM + matching mask ZIP (recommended for prediction + 3D)",
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

    # At least one input must be supplied.
    if uploaded_image is None and uploaded_dicom is None:
        st.warning("Please upload a CT image or a DICOM ZIP.")
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
        display_image = None
        input_array = None
        volume = None
        mask_volume = None
        dicom_error = None
        dicom_status = None
        dicom_slice_index = None

        # --------------------------------------------------------
        # INPUT PRIORITY:
        # DICOM ZIP is used for both prediction and 3D when supplied.
        # Otherwise PNG/JPG is used for prediction.
        # --------------------------------------------------------
        if uploaded_dicom is not None:
            if pydicom is None:
                st.error(
                    "pydicom is not installed. Add pydicom to requirements.txt."
                )
                st.stop()

            volume, mask_volume, dicom_error, dicom_status = load_uploaded_dicom_zip(
                uploaded_dicom
            )

            if dicom_error:
                st.error(dicom_error)
                st.stop()

            if volume is None:
                st.error("DICOM visualization/prediction is not available.")
                st.stop()

            if dicom_status:
                st.caption(f"DICOM study status: {dicom_status}")

            display_image, input_array, dicom_slice_index = (
                prepare_dicom_slice_for_model(volume)
            )

        else:
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

        try:
            save_patient_record(record)
            record_saved = True
        except Exception as save_error:
            # Streamlit Cloud storage can be ephemeral/read-only. Do not let
            # a record-write failure block the prediction or PDF report.
            record_saved = False
            st.warning(
                "Prediction completed, but the patient record could not be "
                f"saved permanently in this deployment: {save_error}"
            )

        if record_saved:
            st.success("Prediction completed and patient report saved.")
        else:
            st.success("Prediction completed.")

        left, right = st.columns(2)

        with left:
            if uploaded_dicom is not None:
                st.image(
                    display_image,
                    caption=f"Representative CT Slice from DICOM (slice {dicom_slice_index + 1})",
                    use_container_width=True,
                )
            else:
                st.image(
                    display_image,
                    caption="Uploaded CT Scan",
                    use_container_width=True,
                )

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
                "a DICOM + matching mask ZIP was not provided."
            )
        elif mask_volume is None:
            if dicom_status:
                st.info(
                    f"{dicom_status} Affected region: Not available."
                )
            else:
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
