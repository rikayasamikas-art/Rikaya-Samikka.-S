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

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    APP_DIR,
    "brain_stroke_model.keras"
)

PATIENT_RECORDS_DIR = os.path.join(
    APP_DIR,
    "patient_records"
)


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
    "CT Prediction + DICOM Mask 3D Visualization"
)


# ============================================================
# SESSION STATE
# ============================================================

if "patient_ready" not in st.session_state:
    st.session_state.patient_ready = False

if "patient_mode" not in st.session_state:
    st.session_state.patient_mode = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

if "ct_prediction" not in st.session_state:
    st.session_state.ct_prediction = None

if "dicom_result" not in st.session_state:
    st.session_state.dicom_result = None

if "dicom_loaded" not in st.session_state:
    st.session_state.dicom_loaded = False


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "brain_stroke_model.keras not found. "
            "Please keep the model file in the same GitHub repository."
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


def predict_image(image):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (224, 224)
    )

    arr = np.array(
        image,
        dtype=np.float32
    )

    arr = arr / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    model = load_model()

    probabilities = model.predict(
        arr,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ] * 100
    )

    return (
        predicted_class,
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

    file_path = os.path.join(
        PATIENT_RECORDS_DIR,
        f"{patient_id}.json"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def load_patient(
    patient_id
):

    file_path = os.path.join(
        PATIENT_RECORDS_DIR,
        f"{patient_id}.json"
    )

    if not os.path.exists(
        file_path
    ):

        return None

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# 1. PATIENT INFORMATION
# ============================================================

st.header(
    "1. Patient Information"
)

patient_type = st.radio(
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

if patient_type == "New Patient":

    col1, col2 = st.columns(2)

    with col1:

        new_name = st.text_input(
            "Patient Name",
            key="new_name"
        )

        new_id = st.text_input(
            "Patient ID",
            key="new_id"
        )

        new_gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ],
            key="new_gender"
        )

    with col2:

        new_address = st.text_area(
            "Address",
            key="new_address"
        )

        new_scan_date = st.date_input(
            "Scan Date",
            value=date.today(),
            key="new_scan_date"
        )

    if st.button(
        "Save New Patient",
        type="primary",
        key="save_new_patient"
    ):

        if not new_name.strip():

            st.warning(
                "Please enter Patient Name."
            )

        elif not new_id.strip():

            st.warning(
                "Please enter Patient ID."
            )

        else:

            patient_id_clean = (
                new_id.strip()
            )

            existing_record = load_patient(
                patient_id_clean
            )

            if existing_record:

                st.warning(
                    "This Patient ID already exists. "
                    "Please select Existing Patient."
                )

            else:

                patient_data = {

                    "name":
                        new_name.strip(),

                    "id":
                        patient_id_clean,

                    "gender":
                        new_gender,

                    "address":
                        new_address.strip(),

                    "scan_date":
                        str(
                            new_scan_date
                        )
                }

                save_patient(
                    patient_id_clean,
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

                st.session_state.dicom_result = None

                st.session_state.dicom_loaded = False

                st.success(
                    "New patient information saved successfully."
                )


# ============================================================
# EXISTING PATIENT
# ============================================================

else:

    existing_id = st.text_input(
        "Enter Existing Patient ID",
        key="existing_patient_id"
    )

    if st.button(
        "Load Patient Information",
        type="primary",
        key="load_existing_patient"
    ):

        if not existing_id.strip():

            st.warning(
                "Please enter Patient ID."
            )

        else:

            record = load_patient(
                existing_id.strip()
            )

            if record:

                st.session_state.patient_ready = True

                st.session_state.patient_mode = (
                    "Existing Patient"
                )

                st.session_state.patient_data = (
                    record
                )

                st.session_state.ct_prediction = None

                st.session_state.dicom_result = None

                st.session_state.dicom_loaded = False

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
# SHOW CURRENT PATIENT
# ============================================================

if (
    st.session_state.patient_ready
    and st.session_state.patient_data
):

    patient = (
        st.session_state.patient_data
    )

    st.subheader(
        "Current Patient Information"
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
    # OLD PATIENT PREVIOUS REPORT
    # --------------------------------------------------------

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

        if patient.get(
            "probabilities"
        ):

            old_prob = patient[
                "probabilities"
            ]

            old_df = pd.DataFrame(
                {
                    "Class": [
                        "Bleeding",
                        "Ischemic",
                        "Normal"
                    ],

                    "Probability (%)": [

                        float(
                            old_prob.get(
                                "Bleeding",
                                0
                            )
                        ),

                        float(
                            old_prob.get(
                                "Ischemic",
                                0
                            )
                        ),

                        float(
                            old_prob.get(
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
# STOP IF PATIENT NOT READY
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
    "Upload a PNG/JPG CT image for AI prediction."
)

uploaded_ct = st.file_uploader(
    "Upload CT Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    key="ct_upload"
)


if uploaded_ct:

    ct_image = Image.open(
        uploaded_ct
    )

    st.image(
        ct_image,
        caption="Uploaded CT Image",
        width=420
    )

    if st.button(
        "Predict Stroke from CT",
        type="primary",
        key="predict_ct"
    ):

        try:

            (
                predicted_class,
                confidence,
                probabilities
            ) = predict_image(
                ct_image
            )

            st.session_state.ct_prediction = {

                "class":
                    predicted_class,

                "confidence":
                    confidence,

                "probabilities":
                    probabilities
            }

            st.success(
                "CT Prediction completed."
            )

        except Exception as error:

            st.error(
                "CT Prediction Error: "
                + str(error)
            )


# ============================================================
# CT RESULT
# ============================================================

if st.session_state.ct_prediction:

    prediction = (
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
# DICOM READER
# ============================================================

def read_dicom(
    file_path
):

    dicom = pydicom.dcmread(
        file_path
    )

    pixels = (
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

    pixels = (
        pixels * slope
        + intercept
    )

    return (
        dicom,
        pixels
    )


# ============================================================
# DICOM ZIP LOADER
# ============================================================

def load_dicom_zip(
    uploaded_zip
):

    temp_dir = tempfile.mkdtemp(
        prefix="brainstroke_dicom_"
    )

    try:

        with zipfile.ZipFile(
            uploaded_zip,
            "r"
        ) as archive:

            base_path = os.path.abspath(
                temp_dir
            )

            for item in archive.infolist():

                destination = os.path.abspath(
                    os.path.join(
                        temp_dir,
                        item.filename
                    )
                )

                if not destination.startswith(
                    base_path + os.sep
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

                    relative_folder = (
                        os.path.relpath(
                            root,
                            temp_dir
                        ).replace(
                            "\\",
                            "/"
                        )
                    )

                    mask_paths[
                        relative_folder
                    ] = full_path


        # --------------------------------------------------------
        # SORT DICOM SLICES
        # --------------------------------------------------------

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

        dicom_paths.sort(
            key=sort_key
        )


        images = []
        dicoms = []
        masks = []

        for dcm_path in dicom_paths:

            try:

                dicom, pixels = (
                    read_dicom(
                        dcm_path
                    )
                )

                images.append(
                    pixels
                )

                dicoms.append(
                    dicom
                )

                relative_folder = (
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
                    mask_paths.get(
                        relative_folder
                    )
                )

                if mask_path:

                    mask_data = np.load(
                        mask_path,
                        allow_pickle=False
                    )

                    mask = mask_data[
                        "mask"
                    ]

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

        matching_masks = sum(
            mask is not None
            for mask in masks
        )

        return (
            images,
            dicoms,
            masks,
            matching_masks,
            temp_dir
        )

    except Exception:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise


# ============================================================
# 3. DICOM + MASK 3D
# ============================================================

st.header(
    "3. DICOM + Mask 3D Visualization"
)

st.caption(
    "Upload the DICOM series with matching mask.npz files. "
    "DICOM is used for 3D visualization only."
)

uploaded_dicom = st.file_uploader(
    "Upload DICOM + Mask ZIP",
    type=["zip"],
    key="dicom_upload"
)


if uploaded_dicom:

    if st.button(
        "Load 3D Visualization",
        type="primary",
        key="load_3d"
    ):

        temp_dir = None

        try:

            with st.spinner(
                "Loading DICOM series and matching masks..."
            ):

                (
                    dicom_images,
                    dicom_objects,
                    dicom_masks,
                    matching_mask_count,
                    temp_dir
                ) = load_dicom_zip(
                    uploaded_dicom
                )

            if not dicom_images:

                st.error(
                    "No readable DICOM slices found in the ZIP."
                )

            else:

                # ----------------------------------------------------
                # SAVE DICOM RESULT
                # ----------------------------------------------------

                st.session_state.dicom_result = {

                    "slices":
                        len(dicom_images),

                    "matching_masks":
                        matching_mask_count,

                    "affected_region":
                        (
                            "Available"
                            if matching_mask_count > 0
                            else "Not available"
                        )
                }

                st.session_state.dicom_loaded = True


                # ----------------------------------------------------
                # DICOM SUMMARY
                # ----------------------------------------------------

                st.success(
                    "DICOM loaded successfully."
                )

                st.write(
                    "**DICOM Slices:** "
                    + str(
                        len(dicom_images)
                    )
                )

                st.write(
                    "**Matching Masks:** "
                    + str(
                        matching_mask_count
                    )
                    + "/"
                    + str(
                        len(dicom_images)
                    )
                )


                # ----------------------------------------------------
                # CLEAN 3D BRAIN
                # ----------------------------------------------------

                volume = np.stack(
                    dicom_images
                )

                downsample_step = max(
                    1,
                    int(
                        max(
                            volume.shape
                        ) / 90
                    )
                )

                volume_small = volume[
                    ::downsample_step,
                    ::downsample_step,
                    ::downsample_step
                ]

                depth, height, width = (
                    volume_small.shape
                )


                # ----------------------------------------------------
                # HEAD REGION
                # ----------------------------------------------------

                grid_y, grid_x = np.ogrid[
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
                        (grid_y - center_y)
                        / radius_y
                    ) ** 2

                    +

                    (
                        (grid_x - center_x)
                        / radius_x
                    ) ** 2

                    <= 1.0
                )


                # ----------------------------------------------------
                # BRAIN SOFT TISSUE
                # ----------------------------------------------------

                brain_volume_mask = (

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


                # ----------------------------------------------------
                # KEEP LARGEST REGION
                # ----------------------------------------------------

                try:

                    from scipy import ndimage

                    labeled_volume, component_count = (
                        ndimage.label(
                            brain_volume_mask
                        )
                    )

                    if component_count > 0:

                        component_sizes = (
                            np.bincount(
                                labeled_volume.ravel()
                            )
                        )

                        component_sizes[0] = 0

                        largest_component = int(
                            np.argmax(
                                component_sizes
                            )
                        )

                        brain_volume_mask = (
                            labeled_volume
                            == largest_component
                        )

                except Exception:

                    pass


                # ----------------------------------------------------
                # BRAIN POINTS
                # ----------------------------------------------------

                z_coords, y_coords, x_coords = (
                    np.where(
                        brain_volume_mask
                    )
                )


                # ----------------------------------------------------
                # LIMIT POINTS
                # ----------------------------------------------------

                if len(
                    x_coords
                ) > 18000:

                    rng = np.random.default_rng(
                        42
                    )

                    point_ids = (
                        rng.choice(
                            len(x_coords),
                            18000,
                            replace=False
                        )
                    )

                    x_coords = x_coords[
                        point_ids
                    ]

                    y_coords = y_coords[
                        point_ids
                    ]

                    z_coords = z_coords[
                        point_ids
                    ]


                # ----------------------------------------------------
                # FIGURE
                # ----------------------------------------------------

                figure = go.Figure()


                # ----------------------------------------------------
                # GRAY BRAIN
                # ----------------------------------------------------

                figure.add_trace(
                    go.Scatter3d(

                        x=x_coords,

                        y=y_coords,

                        z=z_coords,

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
                # MATCHING AFFECTED REGION
                # ----------------------------------------------------

                affected_points = []

                for slice_index, mask in enumerate(
                    dicom_masks
                ):

                    if mask is None:

                        continue

                    small_mask = mask[
                        ::downsample_step,
                        ::downsample_step
                    ]

                    mask_y, mask_x = np.where(
                        small_mask > 0
                    )

                    if len(
                        mask_x
                    ) > 0:

                        points = np.column_stack(
                            [

                                mask_x,

                                mask_y,

                                np.full(
                                    len(mask_x),
                                    slice_index
                                    / downsample_step
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

                        affected_ids = (
                            rng.choice(
                                len(
                                    affected_points
                                ),
                                12000,
                                replace=False
                            )
                        )

                        affected_points = (
                            affected_points[
                                affected_ids
                            ]
                        )


                    # RED AFFECTED REGION

                    figure.add_trace(
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


                # ----------------------------------------------------
                # 3D LAYOUT
                # ----------------------------------------------------

                figure.update_layout(

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
                    figure,
                    use_container_width=True
                )


        except zipfile.BadZipFile:

            st.error(
                "The uploaded file is not a valid ZIP file."
            )

        except Exception as error:

            st.error(
                "DICOM/3D Error: "
                + str(error)
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

dicom_result = (
    st.session_state.dicom_result
)


# ============================================================
# REPORT SUMMARY TABLE
# ============================================================

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
            ],

            [
                "CT Bleeding Probability",
                f"{float(ct_prediction['probabilities'][0] * 100):.2f}%"
            ],

            [
                "CT Ischemic Probability",
                f"{float(ct_prediction['probabilities'][1] * 100):.2f}%"
            ],

            [
                "CT Normal Probability",
                f"{float(ct_prediction['probabilities'][2] * 100):.2f}%"
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
    type="primary",
    key="save_final_report"
):

    final_record = dict(
        patient
    )


    # --------------------------------------------------------
    # CT RESULT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DICOM 3D
    # --------------------------------------------------------

    if dicom_result:

        final_record[
            "dicom_3d"
        ] = dicom_result


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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
        "Final patient report saved successfully."
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

current_y = (
    page_height - 50
)


# ============================================================
# PDF TITLE
# ============================================================

pdf.setFont(
    "Helvetica-Bold",
    16
)

pdf.drawString(
    45,
    current_y,
    "AI-Based Brain Stroke Detection Report"
)

current_y -= 32

pdf.setFont(
    "Helvetica",
    10
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

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


# ============================================================
# CT PDF
# ============================================================

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

else:

    pdf_lines.extend(
        [
            "CT Prediction: Not completed",
            ""
        ]
    )


# ============================================================
# DICOM PDF
# ============================================================

if dicom_result:

    pdf_lines.extend(
        [

            "DICOM 3D: Loaded",

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

else:

    pdf_lines.extend(
        [

            "DICOM 3D: Not loaded",

            "Affected Region: Not available",

            ""
        ]
    )


# ============================================================
# PDF FOOTER TEXT
# ============================================================

pdf_lines.extend(
    [

        "AI-assisted academic project.",

        "This system is not a substitute for clinical diagnosis."
    ]
)


# ============================================================
# DRAW PDF
# ============================================================

for line in pdf_lines:

    if current_y < 60:

        pdf.showPage()

        current_y = (
            page_height - 50
        )

        pdf.setFont(
            "Helvetica",
            10
        )

    pdf.drawString(
        45,
        current_y,
        str(line)[:110]
    )

    current_y -= 17


pdf.save()

pdf_buffer.seek(0)


# ============================================================
# DOWNLOAD PDF
# ============================================================

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


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-assisted academic project. "
    "This system is not a substitute for clinical diagnosis."
)
