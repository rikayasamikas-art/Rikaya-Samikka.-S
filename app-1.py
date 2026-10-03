import os
import io
import json
import shutil
import tempfile
import zipfile
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
PATIENT_RECORDS_DIR = os.path.join(
    PROJECT_DIR,
    "patient_records"
)

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

    arr = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    model = load_model()

    probabilities = model.predict(
        arr,
        verbose=0
    )[0]

    index = int(
        np.argmax(probabilities)
    )

    result = CLASS_NAMES[index]

    confidence = float(
        probabilities[index] * 100
    )

    return (
        result,
        confidence,
        probabilities
    )


# ============================================================
# PATIENT RECORDS
# ============================================================

def save_patient(patient_id, data):

    os.makedirs(
        PATIENT_RECORDS_DIR,
        exist_ok=True
    )

    path = os.path.join(
        PATIENT_RECORDS_DIR,
        f"{patient_id}.json"
    )

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_patient(patient_id):

    path = os.path.join(
        PATIENT_RECORDS_DIR,
        f"{patient_id}.json"
    )

    if os.path.exists(path):

        with open(
            path,
            "r"
        ) as f:

            return json.load(f)

    return None


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("1. Patient Information")

col1, col2 = st.columns(2)

with col1:

    patient_name = st.text_input(
        "Patient Name"
    )

    patient_id = st.text_input(
        "Patient ID"
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

with col2:

    address = st.text_area(
        "Address"
    )

    scan_date = st.date_input(
        "Scan Date",
        value=date.today()
    )


# ============================================================
# EXISTING PATIENT CHECK
# ============================================================

if patient_id:

    old_record = load_patient(
        patient_id
    )

    if old_record:

        st.info(
            "Existing Patient ID found."
        )

        st.json(
            old_record
        )


# ============================================================
# CT PREDICTION
# ============================================================

st.header("2. CT Scan Prediction")

uploaded_image = st.file_uploader(
    "Upload CT Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    key="ct_image_upload"
)


if uploaded_image:

    image = Image.open(
        uploaded_image
    )

    st.image(
        image,
        caption="Uploaded CT Image",
        width=400
    )

    if st.button(
        "Predict Stroke",
        type="primary",
        key="predict_button"
    ):

        try:

            result, confidence, probabilities = (
                predict_image(
                    image
                )
            )

            st.session_state[
                "prediction"
            ] = {

                "class": result,

                "confidence": confidence,

                "probabilities": probabilities
            }

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# RESULT
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state[
        "prediction"
    ]

    st.header(
        "Prediction Result"
    )

    st.success(
        f"Prediction: {prediction['class']}"
    )

    st.metric(
        "Confidence",
        f"{prediction['confidence']:.2f}%"
    )

    probability_df = pd.DataFrame(
        {
            "Class": [
                "Bleeding",
                "Ischemic",
                "Normal"
            ],

            "Probability (%)": [

                float(
                    prediction[
                        "probabilities"
                    ][0] * 100
                ),

                float(
                    prediction[
                        "probabilities"
                    ][1] * 100
                ),

                float(
                    prediction[
                        "probabilities"
                    ][2] * 100
                )
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index(
            "Class"
        )
    )


# ============================================================
# DICOM FUNCTIONS
# ============================================================

def get_studies():

    studies = []

    for split in [
        "test",
        "train",
        "val"
    ]:

        split_path = os.path.join(
            DATASET_DIR,
            split
        )

        if not os.path.exists(
            split_path
        ):
            continue

        for study in os.listdir(
            split_path
        ):

            path = os.path.join(
                split_path,
                study
            )

            if os.path.isdir(path):

                studies.append(
                    (
                        split,
                        study
                    )
                )

    return sorted(
        studies
    )


def read_dicom_file(
    dcm_path
):

    dicom = pydicom.dcmread(
        dcm_path
    )

    image = dicom.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(
            dicom,
            "RescaleSlope",
            1
        )
    )

    intercept = float(
        getattr(
            dicom,
            "RescaleIntercept",
            0
        )
    )

    image = (
        image * slope
        + intercept
    )

    return dicom, image


def load_dicom_study(
    study_path
):

    images = []
    masks = []

    slice_dirs = []

    for slice_name in os.listdir(
        study_path
    ):

        if not slice_name.isdigit():
            continue

        slice_path = os.path.join(
            study_path,
            slice_name
        )

        if os.path.isdir(
            slice_path
        ):

            slice_dirs.append(
                (
                    int(slice_name),
                    slice_path
                )
            )

    for _, slice_path in sorted(
        slice_dirs
    ):

        dcm_path = os.path.join(
            slice_path,
            "raw.dcm"
        )

        mask_path = os.path.join(
            slice_path,
            "mask.npz"
        )

        if not os.path.exists(
            dcm_path
        ):
            continue

        try:

            _, image = read_dicom_file(
                dcm_path
            )

            images.append(
                image
            )

            if os.path.exists(
                mask_path
            ):

                mask = np.load(
                    mask_path,
                    allow_pickle=False
                )["mask"]

                if mask.shape == image.shape:

                    masks.append(
                        mask
                    )

                else:

                    masks.append(
                        None
                    )

            else:

                masks.append(
                    None
                )

        except Exception:
            continue

    return (
        images,
        masks
    )


# ============================================================
# UPLOAD DICOM + MASK ZIP
# ============================================================

def load_uploaded_dicom_zip(
    uploaded_zip
):

    temp_dir = tempfile.mkdtemp(
        prefix="brainstroke_dicom_"
    )

    try:

        with zipfile.ZipFile(
            uploaded_zip,
            "r"
        ) as zf:

            base = os.path.abspath(
                temp_dir
            )

            for member in zf.infolist():

                member_path = os.path.abspath(
                    os.path.join(
                        temp_dir,
                        member.filename
                    )
                )

                if not member_path.startswith(
                    base + os.sep
                ):

                    raise ValueError(
                        "Unsafe ZIP file path detected."
                    )

            zf.extractall(
                temp_dir
            )

        dicom_paths = []

        mask_by_folder = {}

        for root, _, files in os.walk(
            temp_dir
        ):

            for filename in files:

                full_path = os.path.join(
                    root,
                    filename
                )

                lower_name = (
                    filename.lower()
                )

                if lower_name.endswith(
                    ".dcm"
                ):

                    dicom_paths.append(
                        full_path
                    )

                elif lower_name == "mask.npz":

                    relative_dir = os.path.relpath(
                        root,
                        temp_dir
                    ).replace(
                        "\\",
                        "/"
                    )

                    mask_by_folder[
                        relative_dir
                    ] = full_path

        if not dicom_paths:

            return (
                [],
                [],
                0,
                temp_dir
            )

        def sort_key(path):

            parent = os.path.basename(
                os.path.dirname(path)
            )

            if parent.isdigit():

                return (
                    0,
                    int(parent),
                    path
                )

            return (
                1,
                path
            )

        dicom_paths = sorted(
            dicom_paths,
            key=sort_key
        )

        images = []
        masks = []

        mask_count = 0

        for dcm_path in dicom_paths:

            try:

                _, image = (
                    read_dicom_file(
                        dcm_path
                    )
                )

                images.append(
                    image
                )

                relative_dir = (
                    os.path.relpath(
                        os.path.dirname(
                            dcm_path
                        ),
                        temp_dir
                    ).replace(
                        "\\",
                        "/"
                    )
                )

                mask_path = (
                    mask_by_folder.get(
                        relative_dir
                    )
                )

                if mask_path:

                    mask = np.load(
                        mask_path,
                        allow_pickle=False
                    )["mask"]

                    if mask.shape == image.shape:

                        masks.append(
                            mask
                        )

                        mask_count += 1

                    else:

                        masks.append(
                            None
                        )

                else:

                    masks.append(
                        None
                    )

            except Exception:
                continue

        return (
            images,
            masks,
            mask_count,
            temp_dir
        )

    except Exception:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise


# ============================================================
# 3D VISUALIZATION
# ============================================================

st.header(
    "3. DICOM 3D Visualization"
)

st.caption(
    "Upload a ZIP containing the DICOM series and matching mask.npz files."
)

uploaded_dicom_zip = st.file_uploader(
    "Upload DICOM + Mask ZIP",
    type=["zip"],
    key="dicom_mask_zip"
)


# Existing local studies
studies = get_studies()

selected = None
study_path = None

if studies:

    st.subheader(
        "Or Select Existing DICOM Study"
    )

    labels = [

        f"{split.upper()} - {study}"

        for split, study in studies
    ]

    selected = st.selectbox(
        "Select DICOM Study",
        labels,
        key="existing_study_select"
    )

    index = labels.index(
        selected
    )

    split, study = studies[
        index
    ]

    study_path = os.path.join(
        DATASET_DIR,
        split,
        study
    )


# ============================================================
# LOAD 3D BUTTON
# ============================================================

if uploaded_dicom_zip is not None or study_path:

    if st.button(
        "Load 3D Visualization",
        type="primary",
        key="load_3d_button"
    ):

        temp_dir_to_remove = None

        try:

            with st.spinner(
                "Loading DICOM slices and matching masks..."
            ):

                if uploaded_dicom_zip is not None:

                    (
                        images,
                        masks,
                        mask_count,
                        temp_dir_to_remove
                    ) = load_uploaded_dicom_zip(
                        uploaded_dicom_zip
                    )

                    source_text = (
                        "Uploaded DICOM + Mask ZIP"
                    )

                else:

                    (
                        images,
                        masks
                    ) = load_dicom_study(
                        study_path
                    )

                    mask_count = sum(
                        m is not None
                        for m in masks
                    )

                    source_text = (
                        f"Study: {selected}"
                    )

            if not images:

                st.error(
                    "No readable DICOM slices found."
                )

            else:

                st.success(
                    f"{source_text} — "
                    f"Loaded {len(images)} DICOM slices."
                )

                st.write(
                    f"Matching masks: "
                    f"{mask_count}/{len(images)}"
                )

                volume = np.stack(
                    images
                )

                # ====================================================
                # CLEAN 3D BRAIN VISUALIZATION
                # ====================================================

                step = max(
                    1,
                    int(
                        max(
                            volume.shape
                        ) / 90
                    )
                )

                volume_small = volume[
                    ::step,
                    ::step,
                    ::step
                ]

                depth, height, width = (
                    volume_small.shape
                )

                yy_grid, xx_grid = np.ogrid[
                    :height,
                    :width
                ]

                cy = (
                    height - 1
                ) / 2.0

                cx = (
                    width - 1
                ) / 2.0

                ry = (
                    height * 0.46
                )

                rx = (
                    width * 0.46
                )

                head_ellipse = (

                    (
                        (yy_grid - cy)
                        / ry
                    ) ** 2

                    +

                    (
                        (xx_grid - cx)
                        / rx
                    ) ** 2

                    <= 1.0
                )

                # Keep central brain soft tissue
                brain_mask = (

                    (volume_small > -20)

                    &

                    (volume_small < 120)

                    &

                    head_ellipse[
                        None,
                        :,
                        :
                    ]
                )

                # Keep largest connected region
                try:

                    from scipy import ndimage

                    labels_3d, number = (
                        ndimage.label(
                            brain_mask
                        )
                    )

                    if number > 0:

                        sizes = np.bincount(
                            labels_3d.ravel()
                        )

                        sizes[0] = 0

                        largest = int(
                            np.argmax(
                                sizes
                            )
                        )

                        brain_mask = (
                            labels_3d
                            == largest
                        )

                except Exception:
                    pass

                z, y, x = np.where(
                    brain_mask
                )

                # Limit CT points
                if len(x) > 18000:

                    rng = np.random.default_rng(
                        42
                    )

                    selected_points = (
                        rng.choice(
                            len(x),
                            18000,
                            replace=False
                        )
                    )

                    x = x[
                        selected_points
                    ]

                    y = y[
                        selected_points
                    ]

                    z = z[
                        selected_points
                    ]

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter3d(

                        x=x,

                        y=y,

                        z=z,

                        mode="markers",

                        marker=dict(
                            size=1.8,
                            opacity=0.28
                        ),

                        name="Brain CT"
                    )
                )

                # ====================================================
                # AFFECTED REGION
                # ====================================================

                affected_points = []

                for slice_index, mask in enumerate(
                    masks
                ):

                    if mask is None:
                        continue

                    mask_small = mask[
                        ::step,
                        ::step
                    ]

                    yy, xx = np.where(
                        mask_small > 0
                    )

                    if len(xx):

                        points = np.column_stack(
                            [
                                xx,
                                yy,
                                np.full(
                                    len(xx),
                                    slice_index
                                    / step
                                )
                            ]
                        )

                        affected_points.append(
                            points
                        )

                if affected_points:

                    affected_points = np.vstack(
                        affected_points
                    )

                    if len(
                        affected_points
                    ) > 12000:

                        rng = np.random.default_rng(
                            42
                        )

                        ids = rng.choice(
                            len(
                                affected_points
                            ),
                            12000,
                            replace=False
                        )

                        affected_points = (
                            affected_points[
                                ids
                            ]
                        )

                    fig.add_trace(
                        go.Scatter3d(

                            x=affected_points[
                                :, 0
                            ],

                            y=affected_points[
                                :, 1
                            ],

                            z=affected_points[
                                :, 2
                            ],

                            mode="markers",

                            marker=dict(
                                size=3.0,
                                opacity=0.95
                            ),

                            name="Affected Region"
                        )
                    )

                    st.success(
                        "Matching affected region available."
                    )

                else:

                    st.info(
                        "Affected region: Not available"
                    )

                fig.update_layout(

                    title=(
                        "3D Brain CT Visualization"
                    ),

                    height=650,

                    scene=dict(

                        aspectmode="data",

                        xaxis=dict(
                            title="X"
                        ),

                        yaxis=dict(
                            title="Y"
                        ),

                        zaxis=dict(
                            title="Slice"
                        )
                    ),

                    legend=dict(

                        orientation="h",

                        yanchor="bottom",

                        y=1.02,

                        xanchor="left",

                        x=0
                    ),

                    margin=dict(
                        l=0,
                        r=0,
                        t=55,
                        b=0
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except zipfile.BadZipFile:

            st.error(
                "The uploaded file is not a valid ZIP archive."
            )

        except Exception as e:

            st.error(
                f"3D loading error: {e}"
            )

        finally:

            if temp_dir_to_remove:

                shutil.rmtree(
                    temp_dir_to_remove,
                    ignore_errors=True
                )


# ============================================================
# SAVE REPORT
# ============================================================

st.header(
    "4. Save Report"
)

if "prediction" in st.session_state:

    prediction = st.session_state[
        "prediction"
    ]

    patient_data = {

        "name": patient_name,

        "id": patient_id,

        "gender": gender,

        "address": address,

        "scan_date": str(
            scan_date
        ),

        "predicted_class":
            prediction["class"],

        "confidence":
            prediction["confidence"],

        "probabilities": {

            "Bleeding":
                float(
                    prediction[
                        "probabilities"
                    ][0] * 100
                ),

            "Ischemic":
                float(
                    prediction[
                        "probabilities"
                    ][1] * 100
                ),

            "Normal":
                float(
                    prediction[
                        "probabilities"
                    ][2] * 100
                )
        }
    }

    if st.button(
        "Save Patient Report",
        key="save_report_button"
    ):

        if not patient_id:

            st.warning(
                "Please enter Patient ID."
            )

        else:

            save_patient(
                patient_id,
                patient_data
            )

            st.success(
                "Patient report saved successfully."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-assisted academic project. "
    "This system is not a substitute for clinical diagnosis."
)
