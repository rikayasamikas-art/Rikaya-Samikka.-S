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

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# GitHub / Streamlit Cloud
PROJECT_DIR = APP_DIR

# Colab fallback
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/BrainStroke_Project"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "brain_stroke_model.keras"
)

if not os.path.exists(MODEL_PATH):
    drive_model = os.path.join(
        DRIVE_PROJECT_DIR,
        "brain_stroke_model.keras"
    )
    if os.path.exists(drive_model):
        MODEL_PATH = drive_model

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset"
)

if not os.path.exists(DATASET_DIR):
    drive_dataset = os.path.join(
        DRIVE_PROJECT_DIR,
        "dataset"
    )
    if os.path.exists(drive_dataset):
        DATASET_DIR = drive_dataset

PATIENT_RECORDS_DIR = os.path.join(
    PROJECT_DIR,
    "patient_records"
)

if not os.path.exists(PATIENT_RECORDS_DIR):
    drive_records = os.path.join(
        DRIVE_PROJECT_DIR,
        "patient_records"
    )
    if os.path.exists(drive_records):
        PATIENT_RECORDS_DIR = drive_records


# ============================================================
# CLASS MAPPING
# ============================================================

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

st.title(
    "🧠 AI-Based Brain Stroke Detection"
)

st.caption(
    "CT Prediction + DICOM/Mask 3D Visualization"
)


# ============================================================
# SESSION STATE
# ============================================================

default_state = {
    "patient_ready": False,
    "patient_mode": None,
    "patient_data": None,
    "prediction": None,
    "dicom_result": None,
}

for key, value in default_state.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Model file not found: "
            + MODEL_PATH
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


def predict_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

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
# PATIENT RECORD FUNCTIONS
# ============================================================

def save_patient(
    patient_id,
    data
):

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
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_patient(
    patient_id
):

    path = os.path.join(
        PATIENT_RECORDS_DIR,
        f"{patient_id}.json"
    )

    if os.path.exists(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return None


# ============================================================
# 1. PATIENT INFORMATION
# ============================================================

st.header(
    "1. Patient Information"
)

patient_mode = st.radio(
    "Patient Type",
    [
        "New Patient",
        "Existing Patient"
    ],
    horizontal=True
)


# ============================================================
# NEW PATIENT
# ============================================================

if patient_mode == "New Patient":

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

    if st.button(
        "Save New Patient",
        type="primary",
        key="new_patient_save"
    ):

        if not patient_name.strip():

            st.warning(
                "Please enter Patient Name."
            )

        elif not patient_id.strip():

            st.warning(
                "Please enter Patient ID."
            )

        else:

            clean_id = (
                patient_id.strip()
            )

            old_record = load_patient(
                clean_id
            )

            if old_record:

                st.warning(
                    "This Patient ID already exists. "
                    "Please select Existing Patient."
                )

            else:

                patient_data = {

                    "name":
                        patient_name.strip(),

                    "id":
                        clean_id,

                    "gender":
                        gender,

                    "address":
                        address.strip(),

                    "scan_date":
                        str(scan_date)
                }

                save_patient(
                    clean_id,
                    patient_data
                )

                st.session_state.patient_ready = True

                st.session_state.patient_mode = (
                    "New Patient"
                )

                st.session_state.patient_data = (
                    patient_data
                )

                st.session_state.prediction = None

                st.session_state.dicom_result = None

                st.success(
                    "New patient information saved successfully."
                )


# ============================================================
# EXISTING PATIENT
# ============================================================

else:

    old_patient_id = st.text_input(
        "Enter Existing Patient ID",
        key="existing_patient_id"
    )

    if st.button(
        "Load Patient Information",
        type="primary",
        key="existing_patient_load"
    ):

        if not old_patient_id.strip():

            st.warning(
                "Please enter Patient ID."
            )

        else:

            old_record = load_patient(
                old_patient_id.strip()
            )

            if old_record:

                st.session_state.patient_ready = True

                st.session_state.patient_mode = (
                    "Existing Patient"
                )

                st.session_state.patient_data = (
                    old_record
                )

                st.session_state.prediction = None

                st.session_state.dicom_result = None

                st.success(
                    "Existing patient information loaded."
                )

            else:

                st.session_state.patient_ready = False

                st.session_state.patient_data = None

                st.error(
                    "Patient ID not found."
                )


# ============================================================
# SHOW PATIENT INFORMATION
# ============================================================

if (
    st.session_state.patient_ready
    and st.session_state.patient_data
):

    patient = (
        st.session_state.patient_data
    )

    st.subheader(
        "Patient Information"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            "**Patient Name:** "
            + str(
                patient.get(
                    "name",
                    "-"
                )
            )
        )

        st.write(
            "**Patient ID:** "
            + str(
                patient.get(
                    "id",
                    "-"
                )
            )
        )

        st.write(
            "**Gender:** "
            + str(
                patient.get(
                    "gender",
                    "-"
                )
            )
        )

    with c2:

        st.write(
            "**Address:** "
            + str(
                patient.get(
                    "address",
                    "-"
                )
            )
        )

        st.write(
            "**Scan Date:** "
            + str(
                patient.get(
                    "scan_date",
                    "-"
                )
            )
        )

    # --------------------------------------------------------
    # OLD REPORT
    # --------------------------------------------------------

    if (
        st.session_state.patient_mode
        == "Existing Patient"
    ):

        old_class = patient.get(
            "predicted_class"
        )

        old_confidence = patient.get(
            "confidence"
        )

        if old_class:

            st.info(
                "Previous Report: "
                + str(old_class)
                + " - "
                + f"{float(old_confidence):.2f}%"
            )

        old_probabilities = patient.get(
            "probabilities"
        )

        if old_probabilities:

            old_df = pd.DataFrame(
                {
                    "Class": [
                        "Bleeding",
                        "Ischemic",
                        "Normal"
                    ],
                    "Probability (%)": [
                        float(
                            old_probabilities.get(
                                "Bleeding",
                                0
                            )
                        ),
                        float(
                            old_probabilities.get(
                                "Ischemic",
                                0
                            )
                        ),
                        float(
                            old_probabilities.get(
                                "Normal",
                                0
                            )
                        )
                    ]
                }
            )

            st.write(
                "Previous Report Probabilities"
            )

            st.dataframe(
                old_df,
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# DON'T CONTINUE BEFORE PATIENT
# ============================================================

if not st.session_state.patient_ready:

    st.info(
        "Please save a New Patient or load an Existing Patient ID first."
    )

    st.stop()


# ============================================================
# 2. CT SCAN
# ============================================================

st.header(
    "2. CT Scan Prediction"
)

st.caption(
    "Upload CT image for 2D stroke prediction."
)

uploaded_image = st.file_uploader(
    "Upload CT Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    key="ct_upload"
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
        key="predict_stroke"
    ):

        try:

            (
                result,
                confidence,
                probabilities
            ) = predict_image(
                image
            )

            st.session_state.prediction = {

                "class":
                    result,

                "confidence":
                    confidence,

                "probabilities":
                    probabilities
            }

            st.session_state.dicom_result = None

            st.success(
                f"Prediction: {result}"
            )

        except Exception as e:

            st.error(
                "Prediction error: "
                + str(e)
            )


# ============================================================
# SHOW CURRENT PREDICTION
# ============================================================

if st.session_state.prediction:

    prediction = (
        st.session_state.prediction
    )

    st.subheader(
        "CT Prediction Result"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.success(
            "Prediction: "
            + str(
                prediction["class"]
            )
        )

    with c2:

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

def read_dicom_file(
    dcm_path
):

    dicom = pydicom.dcmread(
        dcm_path
    )

    image = (
        dicom.pixel_array.astype(
            np.float32
        )
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

    return (
        dicom,
        image
    )


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

            study_path = os.path.join(
                split_path,
                study
            )

            if os.path.isdir(
                study_path
            ):

                studies.append(
                    (
                        split,
                        study
                    )
                )

    return sorted(
        studies
    )


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
# DICOM ZIP UPLOAD
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
        masks_by_folder = {}

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

                    folder = os.path.relpath(
                        root,
                        temp_dir
                    ).replace(
                        "\\",
                        "/"
                    )

                    masks_by_folder[
                        folder
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
                os.path.dirname(
                    path
                )
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

                folder = os.path.relpath(
                    os.path.dirname(
                        dcm_path
                    ),
                    temp_dir
                ).replace(
                    "\\",
                    "/"
                )

                mask_path = (
                    masks_by_folder.get(
                        folder
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
# 3. DICOM + MASK 3D VISUALIZATION
# ============================================================

st.header(
    "3. DICOM + Mask 3D Visualization"
)

st.caption(
    "DICOM is used for 3D visualization. "
    "Upload the DICOM series with matching mask.npz files."
)

uploaded_dicom_zip = st.file_uploader(
    "Upload DICOM + Mask ZIP",
    type=["zip"],
    key="dicom_zip_upload"
)


# Existing studies
existing_studies = get_studies()

existing_study_path = None
existing_study_label = None

if existing_studies:

    st.subheader(
        "Or Select Existing DICOM Study"
    )

    study_labels = [

        f"{split.upper()} - {study}"

        for split, study in existing_studies
    ]

    existing_study_label = st.selectbox(
        "Select DICOM Study",
        study_labels,
        key="study_select"
    )

    selected_index = study_labels.index(
        existing_study_label
    )

    selected_split, selected_study = (
        existing_studies[
            selected_index
        ]
    )

    existing_study_path = os.path.join(
        DATASET_DIR,
        selected_split,
        selected_study
    )


# ============================================================
# LOAD 3D
# ============================================================

if (
    uploaded_dicom_zip is not None
    or existing_study_path is not None
):

    if st.button(
        "Load 3D Visualization",
        type="primary",
        key="load_3d"
    ):

        temp_dir = None

        try:

            with st.spinner(
                "Loading DICOM slices and matching masks..."
            ):

                if uploaded_dicom_zip is not None:

                    (
                        images,
                        masks,
                        mask_count,
                        temp_dir
                    ) = load_uploaded_dicom_zip(
                        uploaded_dicom_zip
                    )

                    source = (
                        "Uploaded DICOM + Mask ZIP"
                    )

                else:

                    (
                        images,
                        masks
                    ) = load_dicom_study(
                        existing_study_path
                    )

                    mask_count = sum(
                        mask is not None
                        for mask in masks
                    )

                    source = (
                        f"Existing study: "
                        f"{existing_study_label}"
                    )

            if not images:

                st.error(
                    "No readable DICOM slices found."
                )

            else:

                st.success(
                    f"{source} - "
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
                # CLEAN BRAIN VISUALIZATION
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

                _, height, width = (
                    volume_small.shape
                )

                yy_grid, xx_grid = np.ogrid[
                    :height,
                    :width
                ]

                center_y = (
                    height - 1
                ) / 2.0

                center_x = (
                    width - 1
                ) / 2.0

                radius_y = (
                    height * 0.46
                )

                radius_x = (
                    width * 0.46
                )

                head_region = (

                    (
                        (yy_grid - center_y)
                        / radius_y
                    ) ** 2

                    +

                    (
                        (xx_grid - center_x)
                        / radius_x
                    ) ** 2

                    <= 1.0
                )

                # Brain soft tissue
                brain_mask = (

                    (volume_small > -20)

                    &

                    (volume_small < 120)

                    &

                    head_region[
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

                    ids = rng.choice(
                        len(x),
                        18000,
                        replace=False
                    )

                    x = x[ids]
                    y = y[ids]
                    z = z[ids]

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
                # MATCHING AFFECTED REGION
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

                affected_available = (
                    len(affected_points) > 0
                )

                if affected_available:

                    affected_points = (
                        np.vstack(
                            affected_points
                        )
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
                            affected_points[ids]
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

                    margin=dict(
                        l=0,
                        r=0,
                        t=55,
                        b=0
                    ),

                    legend=dict(
                        orientation="h"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Store DICOM result
                st.session_state.dicom_result = {

                    "source":
                        source,

                    "slices":
                        len(images),

                    "matching_masks":
                        mask_count,

                    "affected_region":
                        (
                            "Available"
                            if affected_available
                            else "Not available"
                        )
                }

        except zipfile.BadZipFile:

            st.error(
                "The uploaded file is not a valid ZIP file."
            )

        except Exception as e:

            st.error(
                "3D loading error: "
                + str(e)
            )

        finally:

            if temp_dir:

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )


# ============================================================
# 4. FINAL REPORT
# ============================================================

st.header(
    "4. Final Report"
)

patient = (
    st.session_state.patient_data
)

prediction = (
    st.session_state.prediction
)

dicom_result = (
    st.session_state.dicom_result
)


# ============================================================
# REPORT
# ============================================================

if patient and prediction:

    st.subheader(
        "Report Summary"
    )

    report_rows = [

        [
            "Patient Name",
            patient.get(
                "name",
                "-"
            )
        ],

        [
            "Patient ID",
            patient.get(
                "id",
                "-"
            )
        ],

        [
            "Gender",
            patient.get(
                "gender",
                "-"
            )
        ],

        [
            "Address",
            patient.get(
                "address",
                "-"
            )
        ],

        [
            "Scan Date",
            patient.get(
                "scan_date",
                "-"
            )
        ],

        [
            "CT Prediction",
            prediction[
                "class"
            ]
        ],

        [
            "Confidence",
            f"{prediction['confidence']:.2f}%"
        ],
    ]

    if dicom_result:

        report_rows.extend(
            [

                [
                    "DICOM 3D",
                    "Loaded"
                ],

                [
                    "DICOM Slices",
                    str(
                        dicom_result[
                            "slices"
                        ]
                    )
                ],

                [
                    "Matching Masks",
                    str(
                        dicom_result[
                            "matching_masks"
                        ]
                    )
                ],

                [
                    "Affected Region",
                    dicom_result[
                        "affected_region"
                    ]
                ]
            ]
        )

    else:

        report_rows.extend(
            [

                [
                    "DICOM 3D",
                    "Not loaded"
                ],

                [
                    "Affected Region",
                    "Not available"
                ]
            ]
        )

    report_df = pd.DataFrame(
        report_rows,
        columns=[
            "Field",
            "Value"
        ]
    )

    st.dataframe(
        report_df,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # SAVE FINAL REPORT
    # ========================================================

    if st.button(
        "Save Final Report",
        type="primary",
        key="save_final_report"
    ):

        final_record = {

            "name":
                patient.get(
                    "name",
                    ""
                ),

            "id":
                patient.get(
                    "id",
                    ""
                ),

            "gender":
                patient.get(
                    "gender",
                    ""
                ),

            "address":
                patient.get(
                    "address",
                    ""
                ),

            "scan_date":
                patient.get(
                    "scan_date",
                    ""
                ),

            "predicted_class":
                prediction[
                    "class"
                ],

            "confidence":
                prediction[
                    "confidence"
                ],

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
            },

            "dicom_3d":
                (
                    dicom_result
                    if dicom_result
                    else {
                        "status":
                            "Not loaded"
                    }
                )
        }

        patient_id_for_save = (
            patient.get(
                "id",
                ""
            )
        )

        if patient_id_for_save:

            save_patient(
                patient_id_for_save,
                final_record
            )

            st.session_state.patient_data = (
                final_record
            )

            st.success(
                "Final patient report saved successfully."
            )


    # ========================================================
    # PDF REPORT
    # ========================================================

    pdf_buffer = io.BytesIO()

    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=A4
    )

    page_width, page_height = A4

    y = (
        page_height
        - 50
    )

    pdf.setFont(
        "Helvetica-Bold",
        16
    )

    pdf.drawString(
        50,
        y,
        "AI-Based Brain Stroke Detection Report"
    )

    y -= 35

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf_lines = [

        "Patient Name: "
        + str(
            patient.get(
                "name",
                "-"
            )
        ),

        "Patient ID: "
        + str(
            patient.get(
                "id",
                "-"
            )
        ),

        "Gender: "
        + str(
            patient.get(
                "gender",
                "-"
            )
        ),

        "Address: "
        + str(
            patient.get(
                "address",
                "-"
            )
        ),

        "Scan Date: "
        + str(
            patient.get(
                "scan_date",
                "-"
            )
        ),

        "",

        "CT Prediction: "
        + str(
            prediction[
                "class"
            ]
        ),

        "Confidence: "
        + f"{prediction['confidence']:.2f}%",

        "",

        "Bleeding: "
        + f"{float(prediction['probabilities'][0] * 100):.2f}%",

        "Ischemic: "
        + f"{float(prediction['probabilities'][1] * 100):.2f}%",

        "Normal: "
        + f"{float(prediction['probabilities'][2] * 100):.2f}%",

        ""
    ]


    if dicom_result:

        pdf_lines.extend(
            [

                "DICOM 3D Visualization: Loaded",

                "DICOM Slices: "
                + str(
                    dicom_result[
                        "slices"
                    ]
                ),

                "Matching Masks: "
                + str(
                    dicom_result[
                        "matching_masks"
                    ]
                ),

                "Affected Region: "
                + str(
                    dicom_result[
                        "affected_region"
                    ]
                )
            ]
        )

    else:

        pdf_lines.extend(
            [

                "DICOM 3D Visualization: Not loaded",

                "Affected Region: Not available"
            ]
        )


    pdf_lines.extend(
        [

            "",

            "AI-assisted academic project.",

            "This system is not a substitute for clinical diagnosis."
        ]
    )


    for line in pdf_lines:

        if y < 60:

            pdf.showPage()

            y = (
                page_height
                - 50
            )

            pdf.setFont(
                "Helvetica",
                10
            )

        pdf.drawString(
            50,
            y,
            str(line)
        )

        y -= 17


    pdf.save()

    pdf_buffer.seek(0)


    st.download_button(
        "Download PDF Report",
        data=pdf_buffer,
        file_name=(
            "brain_stroke_report_"
            + str(
                patient.get(
                    "id",
                    "patient"
                )
            )
            + ".pdf"
        ),
        mime="application/pdf",
        key="download_pdf"
    )


else:

    st.info(
        "Complete the CT prediction before generating the final report."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-assisted academic project. "
    "This system is not a substitute for clinical diagnosis."
)
