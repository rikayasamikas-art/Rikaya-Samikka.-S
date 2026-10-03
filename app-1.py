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

PROJECT_DIR = APP_DIR

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "brain_stroke_model.keras"
)

PATIENT_RECORDS_DIR = os.path.join(
    PROJECT_DIR,
    "patient_records"
)

# Optional Colab fallback
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/BrainStroke_Project"

if not os.path.exists(MODEL_PATH):

    fallback_model = os.path.join(
        DRIVE_PROJECT_DIR,
        "brain_stroke_model.keras"
    )

    if os.path.exists(fallback_model):
        MODEL_PATH = fallback_model

if not os.path.exists(PATIENT_RECORDS_DIR):

    fallback_records = os.path.join(
        DRIVE_PROJECT_DIR,
        "patient_records"
    )

    if os.path.exists(fallback_records):
        PATIENT_RECORDS_DIR = fallback_records


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Bleeding",
    1: "Ischemic",
    2: "Normal"
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
    "CT Prediction + DICOM Prediction + 3D Visualization"
)


# ============================================================
# SESSION STATE
# ============================================================

default_state = {
    "patient_ready": False,
    "patient_mode": None,
    "patient_data": None,
    "ct_prediction": None,
    "dicom_prediction": None,
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
            f"Model file not found: {MODEL_PATH}"
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
        type="primary"
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

            existing_record = load_patient(
                clean_id
            )

            if existing_record:

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

                st.session_state.ct_prediction = None

                st.session_state.dicom_prediction = None

                st.session_state.dicom_result = None

                st.success(
                    "New patient information saved successfully."
                )


# ============================================================
# EXISTING PATIENT
# ============================================================

else:

    old_patient_id = st.text_input(
        "Enter Existing Patient ID"
    )

    if st.button(
        "Load Patient Information",
        type="primary"
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

                st.session_state.ct_prediction = None

                st.session_state.dicom_prediction = None

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

    if (
        st.session_state.patient_mode
        == "Existing Patient"
    ):

        if patient.get(
            "predicted_class"
        ):

            st.info(
                "Previous Report: "
                + str(
                    patient.get(
                        "predicted_class"
                    )
                )
                + " | Confidence: "
                + f"{float(patient.get('confidence', 0)):.2f}%"
            )


# ============================================================
# STOP BEFORE PATIENT
# ============================================================

if not st.session_state.patient_ready:

    st.info(
        "Please save a New Patient or load an Existing Patient ID first."
    )

    st.stop()


# ============================================================
# 2. CT SCAN PREDICTION
# ============================================================

st.header(
    "2. CT Scan Prediction"
)

st.caption(
    "Upload CT image for 2D AI prediction."
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

    ct_image = Image.open(
        uploaded_image
    )

    st.image(
        ct_image,
        caption="Uploaded CT Image",
        width=420
    )

    if st.button(
        "Predict Stroke from CT",
        type="primary"
    ):

        try:

            (
                result,
                confidence,
                probabilities
            ) = predict_image(
                ct_image
            )

            st.session_state.ct_prediction = {

                "class":
                    result,

                "confidence":
                    confidence,

                "probabilities":
                    probabilities
            }

            st.success(
                f"CT Prediction: {result}"
            )

        except Exception as e:

            st.error(
                f"CT Prediction Error: {e}"
            )


# ============================================================
# CT RESULT
# ============================================================

if st.session_state.ct_prediction:

    pred = (
        st.session_state.ct_prediction
    )

    st.subheader(
        "CT Prediction Result"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.success(
            "Stroke Type: "
            + str(
                pred["class"]
            )
        )

    with c2:

        st.metric(
            "Confidence",
            f"{pred['confidence']:.2f}%"
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
                    pred[
                        "probabilities"
                    ][0] * 100
                ),

                float(
                    pred[
                        "probabilities"
                    ][1] * 100
                ),

                float(
                    pred[
                        "probabilities"
                    ][2] * 100
                )
            ]
        }
    )

    st.dataframe(
        probability_df,
        hide_index=True,
        use_container_width=True
    )

    st.bar_chart(
        probability_df.set_index(
            "Class"
        )
    )


# ============================================================
# DICOM READING
# ============================================================

def read_dicom_file(
    path
):

    dcm = pydicom.dcmread(
        path
    )

    pixels = (
        dcm.pixel_array.astype(
            np.float32
        )
    )

    slope = float(
        getattr(
            dcm,
            "RescaleSlope",
            1
        )
    )

    intercept = float(
        getattr(
            dcm,
            "RescaleIntercept",
            0
        )
    )

    pixels = (
        pixels * slope
        + intercept
    )

    return (
        dcm,
        pixels
    )


# ============================================================
# DICOM TO MODEL IMAGE
# ============================================================

def dicom_to_model_array(
    pixels,
    dcm=None
):

    center = 40.0
    width = 80.0

    if dcm is not None:

        try:

            wc = getattr(
                dcm,
                "WindowCenter",
                center
            )

            ww = getattr(
                dcm,
                "WindowWidth",
                width
            )

            if isinstance(
                wc,
                pydicom.multival.MultiValue
            ):

                wc = wc[0]

            if isinstance(
                ww,
                pydicom.multival.MultiValue
            ):

                ww = ww[0]

            center = float(wc)

            width = float(ww)

            if width <= 1:

                width = 80.0

        except Exception:

            pass

    low = (
        center
        - width / 2.0
    )

    high = (
        center
        + width / 2.0
    )

    clipped = np.clip(
        pixels,
        low,
        high
    )

    image8 = (
        (
            clipped - low
        )
        / (
            high - low
        )
        * 255.0
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        image8
    ).convert(
        "RGB"
    )

    image = image.resize(
        (224, 224)
    )

    return (
        np.array(
            image,
            dtype=np.float32
        )
        / 255.0
    )


# ============================================================
# DICOM PREDICTION
# ============================================================

def predict_dicom(
    images,
    dcms,
    maximum_slices=40
):

    if not images:

        raise ValueError(
            "No DICOM slices found."
        )

    total = len(images)

    if total > maximum_slices:

        indices = np.linspace(
            0,
            total - 1,
            maximum_slices,
            dtype=int
        )

    else:

        indices = np.arange(
            total
        )

    batch_images = []

    for i in indices:

        batch_images.append(
            dicom_to_model_array(
                images[i],
                dcms[i]
            )
        )

    batch = np.stack(
        batch_images
    )

    model = load_model()

    slice_probabilities = (
        model.predict(
            batch,
            verbose=0
        )
    )

    study_probabilities = (
        np.mean(
            slice_probabilities,
            axis=0
        )
    )

    index = int(
        np.argmax(
            study_probabilities
        )
    )

    result = CLASS_NAMES[
        index
    ]

    confidence = float(
        study_probabilities[
            index
        ] * 100
    )

    return (
        result,
        confidence,
        study_probabilities,
        len(indices)
    )


# ============================================================
# ZIP LOADER
# ============================================================

def load_dicom_zip(
    uploaded_zip
):

    temp_dir = tempfile.mkdtemp(
        prefix="brainstroke_"
    )

    try:

        with zipfile.ZipFile(
            uploaded_zip,
            "r"
        ) as archive:

            base = os.path.abspath(
                temp_dir
            )

            for member in archive.infolist():

                target = os.path.abspath(
                    os.path.join(
                        temp_dir,
                        member.filename
                    )
                )

                if not target.startswith(
                    base + os.sep
                ):

                    raise ValueError(
                        "Unsafe ZIP path detected."
                    )

            archive.extractall(
                temp_dir
            )

        dicom_paths = []

        mask_paths = {}

        for root, _, files in os.walk(
            temp_dir
        ):

            for filename in files:

                full_path = os.path.join(
                    root,
                    filename
                )

                lower = filename.lower()

                if lower.endswith(
                    ".dcm"
                ):

                    dicom_paths.append(
                        full_path
                    )

                elif lower == "mask.npz":

                    key = os.path.relpath(
                        root,
                        temp_dir
                    ).replace(
                        "\\",
                        "/"
                    )

                    mask_paths[
                        key
                    ] = full_path

        def sort_key(
            path
        ):

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
        dcms = []
        masks = []

        for dcm_path in dicom_paths:

            try:

                dcm, pixels = (
                    read_dicom_file(
                        dcm_path
                    )
                )

                images.append(
                    pixels
                )

                dcms.append(
                    dcm
                )

                key = os.path.relpath(
                    os.path.dirname(
                        dcm_path
                    ),
                    temp_dir
                ).replace(
                    "\\",
                    "/"
                )

                mask_path = (
                    mask_paths.get(
                        key
                    )
                )

                if mask_path:

                    mask = np.load(
                        mask_path,
                        allow_pickle=False
                    )["mask"]

                    if mask.shape == pixels.shape:

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

        mask_count = sum(
            m is not None
            for m in masks
        )

        return (
            images,
            masks,
            dcms,
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
# 3. DICOM + MASK
# ============================================================

st.header(
    "3. DICOM + Mask Prediction & 3D Visualization"
)

st.caption(
    "Upload DICOM ZIP. The same CNN model predicts the DICOM study, "
    "and matching mask.npz files are used for the 3D affected region."
)

uploaded_dicom = st.file_uploader(
    "Upload DICOM + Mask ZIP",
    type=["zip"],
    key="dicom_zip"
)


if uploaded_dicom:

    if st.button(
        "Predict DICOM + Load 3D",
        type="primary"
    ):

        temp_dir = None

        try:

            with st.spinner(
                "Loading DICOM, predicting stroke type, and preparing 3D..."
            ):

                (
                    images,
                    masks,
                    dcms,
                    mask_count,
                    temp_dir
                ) = load_dicom_zip(
                    uploaded_dicom
                )

            if not images:

                st.error(
                    "No readable DICOM slices found in the ZIP."
                )

            else:

                (
                    d_class,
                    d_confidence,
                    d_probabilities,
                    used_slices
                ) = predict_dicom(
                    images,
                    dcms
                )

                st.session_state.dicom_prediction = {

                    "class":
                        d_class,

                    "confidence":
                        d_confidence,

                    "probabilities":
                        d_probabilities,

                    "used_slices":
                        used_slices
                }

                st.session_state.dicom_result = {

                    "slices":
                        len(images),

                    "matching_masks":
                        mask_count,

                    "affected_region":
                        (
                            "Available"
                            if mask_count > 0
                            else "Not available"
                        )
                }

                # ----------------------------------------------------
                # DICOM PREDICTION
                # ----------------------------------------------------

                st.subheader(
                    "DICOM Study Prediction"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.success(
                        "Stroke Type: "
                        + str(
                            d_class
                        )
                    )

                with c2:

                    st.metric(
                        "Confidence",
                        f"{d_confidence:.2f}%"
                    )

                st.caption(
                    "DICOM prediction is calculated from sampled DICOM slices."
                )

                dicom_probability_df = pd.DataFrame(
                    {

                        "Class": [
                            "Bleeding",
                            "Ischemic",
                            "Normal"
                        ],

                        "Probability (%)": [

                            float(
                                d_probabilities[
                                    0
                                ] * 100
                            ),

                            float(
                                d_probabilities[
                                    1
                                ] * 100
                            ),

                            float(
                                d_probabilities[
                                    2
                                ] * 100
                            )
                        ]
                    }
                )

                st.dataframe(
                    dicom_probability_df,
                    hide_index=True,
                    use_container_width=True
                )

                st.bar_chart(
                    dicom_probability_df.set_index(
                        "Class"
                    )
                )

                st.write(
                    "**DICOM Slices:** "
                    + str(
                        len(images)
                    )
                )

                st.write(
                    "**Slices Used for Prediction:** "
                    + str(
                        used_slices
                    )
                )

                st.write(
                    "**Matching Masks:** "
                    + str(
                        mask_count
                    )
                    + "/"
                    + str(
                        len(images)
                    )
                )

                # ----------------------------------------------------
                # CLEAN 3D BRAIN
                # ----------------------------------------------------

                volume = np.stack(
                    images
                )

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

                _,
                height,
                width = (
                    volume_small.shape
                )

                yy_grid, xx_grid = (
                    np.ogrid[
                        :height,
                        :width
                    ]
                )

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

                head_region = (

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

                # Brain soft-tissue range
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

                # Largest connected component
                try:

                    from scipy import ndimage

                    labels_3d, count = (
                        ndimage.label(
                            brain_mask
                        )
                    )

                    if count > 0:

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

                # Limit points
                if len(x) > 18000:

                    rng = np.random.default_rng(
                        42
                    )

                    ids = rng.choice(
                        len(x),
                        18000,
                        replace=False
                    )

                    x = x[
                        ids
                    ]

                    y = y[
                        ids
                    ]

                    z = z[
                        ids
                    ]

                fig = go.Figure()

                # Gray brain
                fig.add_trace(
                    go.Scatter3d(

                        x=x,

                        y=y,

                        z=z,

                        mode="markers",

                        marker=dict(
                            size=1.8,
                            opacity=0.28,
                            color="lightgray"
                        ),

                        name="Brain CT"
                    )
                )

                # ----------------------------------------------------
                # RED AFFECTED REGION
                # ----------------------------------------------------

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
                                opacity=0.95,
                                color="red"
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

        except zipfile.BadZipFile:

            st.error(
                "The uploaded file is not a valid ZIP file."
            )

        except Exception as e:

            st.error(
                "DICOM/3D Error: "
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

ct_prediction = (
    st.session_state.ct_prediction
)

dicom_prediction = (
    st.session_state.dicom_prediction
)

dicom_result = (
    st.session_state.dicom_result
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
    ]
]


# ============================================================
# CT REPORT
# ============================================================

if ct_prediction:

    report_rows.extend(
        [

            [
                "CT Stroke Type",
                ct_prediction[
                    "class"
                ]
            ],

            [
                "CT Confidence",
                f"{ct_prediction['confidence']:.2f}%"
            ]
        ]
    )

else:

    report_rows.append(
        [
            "CT Prediction",
            "Not completed"
        ]
    )


# ============================================================
# DICOM REPORT
# ============================================================

if dicom_prediction:

    report_rows.extend(
        [

            [
                "DICOM Stroke Type",
                dicom_prediction[
                    "class"
                ]
            ],

            [
                "DICOM Confidence",
                f"{dicom_prediction['confidence']:.2f}%"
            ]
        ]
    )

else:

    report_rows.append(
        [
            "DICOM Prediction",
            "Not completed"
        ]
    )


if dicom_result:

    report_rows.extend(
        [

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

    report_rows.append(
        [
            "DICOM 3D",
            "Not loaded"
        ]
    )


st.dataframe(
    pd.DataFrame(
        report_rows,
        columns=[
            "Field",
            "Value"
        ]
    ),
    hide_index=True,
    use_container_width=True
)


# ============================================================
# SAVE FINAL REPORT
# ============================================================

if st.button(
    "Save Final Report",
    type="primary"
):

    final_record = dict(
        patient
    )

    if ct_prediction:

        final_record[
            "predicted_class"
        ] = ct_prediction[
            "class"
        ]

        final_record[
            "confidence"
        ] = ct_prediction[
            "confidence"
        ]

        final_record[
            "probabilities"
        ] = {

            "Bleeding":
                float(
                    ct_prediction[
                        "probabilities"
                    ][0] * 100
                ),

            "Ischemic":
                float(
                    ct_prediction[
                        "probabilities"
                    ][1] * 100
                ),

            "Normal":
                float(
                    ct_prediction[
                        "probabilities"
                    ][2] * 100
                )
        }

    if dicom_prediction:

        final_record[
            "dicom_predicted_class"
        ] = dicom_prediction[
            "class"
        ]

        final_record[
            "dicom_confidence"
        ] = dicom_prediction[
            "confidence"
        ]

        final_record[
            "dicom_probabilities"
        ] = {

            "Bleeding":
                float(
                    dicom_prediction[
                        "probabilities"
                    ][0] * 100
                ),

            "Ischemic":
                float(
                    dicom_prediction[
                        "probabilities"
                    ][1] * 100
                ),

            "Normal":
                float(
                    dicom_prediction[
                        "probabilities"
                    ][2] * 100
                )
        }

    if dicom_result:

        final_record[
            "dicom_3d"
        ] = dicom_result

    save_patient(
        patient.get(
            "id",
            "patient"
        ),
        final_record
    )

    st.session_state.patient_data = (
        final_record
    )

    st.success(
        "Final report saved successfully."
    )


# ============================================================
# PDF REPORT
# ============================================================

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
    45,
    y,
    "AI-Based Brain Stroke Detection Report"
)

y -= 32

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

    ""
]


if ct_prediction:

    pdf_lines.extend(
        [

            "CT Stroke Type: "
            + str(
                ct_prediction[
                    "class"
                ]
            ),

            "CT Confidence: "
            + f"{ct_prediction['confidence']:.2f}%",

            "CT Bleeding: "
            + f"{float(ct_prediction['probabilities'][0] * 100):.2f}%",

            "CT Ischemic: "
            + f"{float(ct_prediction['probabilities'][1] * 100):.2f}%",

            "CT Normal: "
            + f"{float(ct_prediction['probabilities'][2] * 100):.2f}%",

            ""
        ]
    )


if dicom_prediction:

    pdf_lines.extend(
        [

            "DICOM Stroke Type: "
            + str(
                dicom_prediction[
                    "class"
                ]
            ),

            "DICOM Confidence: "
            + f"{dicom_prediction['confidence']:.2f}%",

            "DICOM Bleeding: "
            + f"{float(dicom_prediction['probabilities'][0] * 100):.2f}%",

            "DICOM Ischemic: "
            + f"{float(dicom_prediction['probabilities'][1] * 100):.2f}%",

            "DICOM Normal: "
            + f"{float(dicom_prediction['probabilities'][2] * 100):.2f}%",

            ""
        ]
    )


if dicom_result:

    pdf_lines.extend(
        [

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
            ),

            ""
        ]
    )


pdf_lines.extend(
    [

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
        45,
        y,
        str(line)[:110]
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
    mime="application/pdf"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-assisted academic project. "
    "This system is not a substitute for clinical diagnosis."
)
