
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import hashlib
import json
import time
import textwrap

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MRI Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "MRI_EfficientNetB0_FINAL.keras"
)

IMG_SIZE = 224
THRESHOLD = 0.05
IG_STEPS = 32


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        margin-top: 10px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array /= 255.0

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict(image_array):

    batch = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(
            batch,
            verbose=0
        )[0][0]
    )

    prediction = (
        1
        if probability >= THRESHOLD
        else 0
    )

    return probability, prediction


# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

def compute_integrated_gradients(
    image_array,
    steps=32
):

    input_tensor = tf.convert_to_tensor(
        image_array[None, ...],
        dtype=tf.float32
    )

    baseline = tf.zeros_like(
        input_tensor
    )

    difference = (
        input_tensor - baseline
    )

    total_gradients = tf.zeros_like(
        input_tensor
    )

    for step in range(
        1,
        steps + 1
    ):

        alpha = tf.cast(
            step / steps,
            tf.float32
        )

        interpolated = (
            baseline +
            alpha * difference
        )

        with tf.GradientTape() as tape:

            tape.watch(
                interpolated
            )

            output = model(
                interpolated,
                training=False
            )

            score = output[:, 0]

        gradients = tape.gradient(
            score,
            interpolated
        )

        if gradients is not None:

            total_gradients += gradients

    avg_gradients = (
        total_gradients /
        steps
    )

    attribution = (
        difference *
        avg_gradients
    )

    attribution = attribution[0].numpy()

    importance = np.mean(
        np.abs(attribution),
        axis=-1
    )

    importance = np.nan_to_num(
        importance
    )

    if np.max(importance) > 0:

        importance = (
            importance /
            np.max(importance)
        )

    importance = cv2.GaussianBlur(
        importance.astype(np.float32),
        (0, 0),
        2
    )

    return importance

# ============================================================
# CASE ANALYTICS HELPERS
# ============================================================

def get_case_analytics(original_image, image_array):
    """Calculate descriptive statistics for the uploaded image."""

    arr = np.asarray(
        original_image.convert("RGB")
    )

    processed = np.clip(
        image_array * 255.0,
        0,
        255
    ).astype(np.uint8)

    gray = cv2.cvtColor(
        processed,
        cv2.COLOR_RGB2GRAY
    )

    gray_norm = (
        gray.astype(np.float32) / 255.0
    )

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    total = float(hist.sum())

    if total > 0:
        p = hist / total
        p = p[p > 0]
        entropy = float(
            -np.sum(p * np.log2(p))
        )
    else:
        entropy = 0.0

    width, height = original_image.size

    return {
        "width": int(width),
        "height": int(height),
        "processed_width": IMG_SIZE,
        "processed_height": IMG_SIZE,
        "aspect_ratio": (
            width / height
            if height else 0.0
        ),
        "mean": float(np.mean(image_array)),
        "std": float(np.std(image_array)),
        "minimum": float(np.min(image_array)),
        "maximum": float(np.max(image_array)),
        "brightness": float(np.mean(gray_norm)),
        "contrast": float(np.std(gray_norm)),
        "sharpness": float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        ),
        "entropy": entropy
    }


def probability_graph(probability, threshold):

    fig, ax = plt.subplots(
        figsize=(9, 1.8)
    )

    ax.barh(
        [0],
        [1],
        height=0.35,
        alpha=0.12
    )

    ax.barh(
        [0],
        [probability],
        height=0.35,
        alpha=0.85
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2%}"
    )

    ax.axvline(
        probability,
        linewidth=2,
        label=f"P(Tumor) = {probability:.2%}"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Tumor probability"
    )

    ax.set_title(
        "Prediction Probability vs Decision Threshold"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2
    )

    fig.tight_layout()

    return fig


def intensity_histogram(image_array):

    gray = cv2.cvtColor(
        np.clip(
            image_array * 255,
            0,
            255
        ).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.hist(
        gray.ravel(),
        bins=64,
        alpha=0.85
    )

    ax.set_title(
        "MRI Pixel Intensity Histogram"
    )

    ax.set_xlabel(
        "Pixel intensity (0–255)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def intensity_distribution(image_array):

    gray = cv2.cvtColor(
        np.clip(
            image_array * 255,
            0,
            255
        ).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ) / 255.0

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.boxplot(
        gray.ravel()
    )

    ax.set_title(
        "Normalized Pixel Distribution"
    )

    ax.set_ylabel(
        "Normalized intensity"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def build_case_report(
    filename,
    probability,
    prediction,
    analytics,
    attr_summary
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    case_id = hashlib.sha256(
        f"{filename}|{timestamp}".encode()
    ).hexdigest()[:12].upper()

    result = (
        "Tumor"
        if prediction == 1
        else "No Tumor"
    )

    return f"""
MRI TUMOR AI — INDIVIDUAL CASE ANALYSIS
========================================

CASE INFORMATION
----------------
Case ID              : {case_id}
Analysis time        : {timestamp}
File                 : {filename}
Model                : MRI_EfficientNetB0

AI RESULT
---------
Prediction           : {result}
Tumor probability    : {probability:.2%}
Decision threshold   : {THRESHOLD:.2%}
Threshold status     : {"ABOVE" if probability >= THRESHOLD else "BELOW"}

IMAGE INFORMATION
-----------------
Original dimensions  : {analytics["width"]} x {analytics["height"]} px
Processed dimensions : {analytics["processed_width"]} x {analytics["processed_height"]} px
Aspect ratio         : {analytics["aspect_ratio"]:.3f}
Mean intensity       : {analytics["mean"]:.6f}
Intensity SD         : {analytics["std"]:.6f}
Minimum intensity    : {analytics["minimum"]:.6f}
Maximum intensity    : {analytics["maximum"]:.6f}
Brightness indicator : {analytics["brightness"]:.6f}
Contrast indicator   : {analytics["contrast"]:.6f}
Sharpness indicator  : {analytics["sharpness"]:.3f}
Entropy              : {analytics["entropy"]:.6f}

EXPLAINABILITY
--------------
Method               : Integrated Gradients
Steps                : {IG_STEPS}
Attribution mean     : {attr_summary["mean"]:.8e}
Attribution minimum  : {attr_summary["min"]:.8e}
Attribution maximum  : {attr_summary["max"]:.8e}
Positive attribution : {attr_summary["positive_fraction"]:.2%}

INDEPENDENT TEST PERFORMANCE
----------------------------
Accuracy             : 91.89%
Precision            : 95.98%
Sensitivity/Recall   : 83.50%
Specificity          : 97.61%
F1-score             : 89.30%
ROC-AUC              : 97.46%

RESEARCH / SAFETY NOTICE
------------------------
This application is an academic/research prototype.
The model output is not a clinical diagnosis.
Attribution maps are not tumor segmentation,
tumor-size measurement, tumor grading, or treatment guidance.
"""


# ============================================================
# CONFUSION MATRIX
# ============================================================

def show_confusion_matrix():

    cm = np.array([
        [286, 7],
        [33, 167]
    ])

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        cm
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["No Tumor", "Tumor"]
    )

    ax.set_yticklabels(
        ["No Tumor", "Tumor"]
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_title(
        "Independent Test Confusion Matrix"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold"
            )

    fig.colorbar(
        im,
        ax=ax
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 MRI Tumor AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Assisted Brain MRI Tumor Classification '
    'and Explainability System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        "**Architecture**"
    )
    st.write(
        "EfficientNetB0"
    )

    st.write(
        "**Input Size**"
    )
    st.write(
        "224 × 224 × 3"
    )

    st.write(
        "**Classifier**"
    )
    st.write(
        "Binary Tumor Classification"
    )

    st.write(
        "**Decision Threshold**"
    )
    st.write(
        "0.05"
    )

    st.divider()

    st.header("📊 Test Performance")

    st.metric(
        "Accuracy",
        "91.89%"
    )

    st.metric(
        "Precision",
        "95.98%"
    )

    st.metric(
        "Sensitivity / Recall",
        "83.50%"
    )

    st.metric(
        "Specificity",
        "97.61%"
    )

    st.metric(
        "F1-score",
        "89.30%"
    )

    st.metric(
        "ROC-AUC",
        "97.46%"
    )

    st.divider()

    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# TABS
# ============================================================

tab_detection, tab_performance, tab_explain, tab_about = st.tabs(
    [
        "🔬 Detection",
        "📊 Model Performance",
        "🔍 Explainability",
        "ℹ️ About"
    ]
)


# ============================================================
# DETECTION TAB
# ============================================================

with tab_detection:

    st.subheader(
        "Brain MRI Tumor Detection"
    )

    st.write(
        "Upload a brain MRI image to obtain an "
        "AI-generated tumor probability and classification."
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = preprocess_image(
            original_image
        )

        probability, prediction = predict(
            image_array
        )

        left, right = st.columns(
            [1, 1]
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with left:

            st.markdown(
                "### 🖼️ Input MRI"
            )

            st.image(
                original_image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with right:

            st.markdown(
                "### 🧠 AI Result"
            )

            if prediction == 1:

                st.error(
                    "⚠️ TUMOR DETECTED"
                )

                result_text = "Tumor"

            else:

                st.success(
                    "✓ NO TUMOR DETECTED"
                )

                result_text = "No Tumor"

            st.metric(
                "Tumor Probability",
                f"{probability:.2%}"
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )

            st.write(
                f"**Prediction:** {result_text}"
            )

            st.write(
                f"**Decision threshold:** "
                f"{THRESHOLD:.2f}"
            )

        st.divider()

        st.info(
            "The prediction is generated by the trained "
            "EfficientNetB0 model using a threshold selected "
            "from the validation dataset."
        )

        # Store for explainability
        st.session_state["image_array"] = image_array
        st.session_state["original_image"] = original_image
        st.session_state["probability"] = probability
        st.session_state["prediction"] = prediction
        st.session_state["filename"] = uploaded_file.name

    else:

        st.info(
            "👆 Upload a brain MRI image to begin analysis."
        )


# ============================================================
# PERFORMANCE TAB
# ============================================================

with tab_performance:

    st.subheader(
        "Independent Test Performance"
    )

    st.write(
        "Final performance of the frozen EfficientNetB0 "
        "model on the independent test set."
    )

    # Metrics
    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Accuracy",
            "91.89%"
        )

    with c2:

        st.metric(
            "Precision",
            "95.98%"
        )

    with c3:

        st.metric(
            "Recall",
            "83.50%"
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        st.metric(
            "Specificity",
            "97.61%"
        )

    with c5:

        st.metric(
            "F1-score",
            "89.30%"
        )

    with c6:

        st.metric(
            "ROC-AUC",
            "97.46%"
        )

    st.divider()

    st.subheader(
        "Confusion Matrix"
    )

    show_confusion_matrix()

    st.divider()

    st.subheader(
        "Test Set Summary"
    )

    summary_data = {
        "Measure": [
            "Test Samples",
            "No Tumor",
            "Tumor",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives"
        ],
        "Value": [
            "493",
            "293",
            "200",
            "286",
            "7",
            "33",
            "167"
        ]
    }

    st.table(
        summary_data
    )

    st.caption(
        "Threshold used for final test evaluation: 0.05"
    )



# ============================================================
# EXPLAINABILITY TAB
# ============================================================

with tab_explain:

    st.subheader(
        "🔍 Model Explainability & Case Analysis"
    )

    st.write(
        "Generate an image-specific explanation, "
        "image analytics and a case summary."
    )

    if "image_array" not in st.session_state:

        st.info(
            "First upload an MRI in the Detection tab."
        )

    else:

        image_array = st.session_state[
            "image_array"
        ]

        original_image = st.session_state[
            "original_image"
        ]

        probability = float(
            st.session_state[
                "probability"
            ]
        )

        prediction = int(
            st.session_state[
                "prediction"
            ]
        )

        filename = st.session_state.get(
            "filename",
            "uploaded_mri"
        )

        # ====================================================
        # IMAGE INFORMATION
        # ====================================================

        analytics = get_case_analytics(
            original_image,
            image_array
        )

        st.markdown(
            "### 🧾 Image Information"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Original Size",
                f'{analytics["width"]} × '
                f'{analytics["height"]}'
            )

        with c2:
            st.metric(
                "Processed Size",
                f'{IMG_SIZE} × {IMG_SIZE}'
            )

        with c3:
            st.metric(
                "Aspect Ratio",
                f'{analytics["aspect_ratio"]:.3f}'
            )

        with c4:
            st.metric(
                "Mean Intensity",
                f'{analytics["mean"]:.4f}'
            )

        c5, c6, c7, c8 = st.columns(4)

        with c5:
            st.metric(
                "Intensity SD",
                f'{analytics["std"]:.4f}'
            )

        with c6:
            st.metric(
                "Brightness",
                f'{analytics["brightness"]:.4f}'
            )

        with c7:
            st.metric(
                "Contrast",
                f'{analytics["contrast"]:.4f}'
            )

        with c8:
            st.metric(
                "Entropy",
                f'{analytics["entropy"]:.4f}'
            )

        st.caption(
            "These are descriptive image statistics only; "
            "they are not clinical measurements."
        )

        # ====================================================
        # IMAGE QUALITY INDICATORS
        # ====================================================

        st.markdown(
            "### 🧪 Image Quality Indicators"
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.metric(
                "Sharpness Indicator",
                f'{analytics["sharpness"]:.2f}'
            )

        with q2:
            st.metric(
                "Intensity Range",
                f'{analytics["minimum"]:.3f} – '
                f'{analytics["maximum"]:.3f}'
            )

        with q3:
            st.metric(
                "Color Mode",
                original_image.mode
            )

        # ====================================================
        # LIVE PROBABILITY GRAPH
        # ====================================================

        st.markdown(
            "### 📈 Live Prediction Analysis"
        )

        prob_fig = probability_graph(
            probability,
            THRESHOLD
        )

        st.pyplot(
            prob_fig,
            use_container_width=True
        )

        plt.close(
            prob_fig
        )

        st.info(
            f"P(Tumor) = {probability:.2%}  |  "
            f"Threshold = {THRESHOLD:.2%}  |  "
            f"Prediction = "
            f"{'Tumor' if prediction == 1 else 'No Tumor'}"
        )

        # ====================================================
        # IMAGE GRAPHS
        # ====================================================

        st.markdown(
            "### 📊 Uploaded MRI Analytics"
        )

        graph1, graph2 = st.columns(2)

        with graph1:

            hist_fig = intensity_histogram(
                image_array
            )

            st.pyplot(
                hist_fig,
                use_container_width=True
            )

            plt.close(
                hist_fig
            )

        with graph2:

            dist_fig = intensity_distribution(
                image_array
            )

            st.pyplot(
                dist_fig,
                use_container_width=True
            )

            plt.close(
                dist_fig
            )

        # ====================================================
        # EXISTING EXPLANATION
        # ====================================================

        st.markdown(
            "### 🔬 Integrated Gradients Explanation"
        )

        generate = st.button(
            "🔍 Generate Explanation",
            type="primary"
        )

        if generate:

            start_time = time.perf_counter()

            with st.spinner(
                "Computing Integrated Gradients..."
            ):

                attribution = (
                    compute_integrated_gradients(
                        image_array,
                        IG_STEPS
                    )
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            attr = np.asarray(
                attribution,
                dtype=np.float32
            )

            attr_summary = {
                "mean": float(
                    np.mean(attr)
                ),
                "min": float(
                    np.min(attr)
                ),
                "max": float(
                    np.max(attr)
                ),
                "positive_fraction": float(
                    np.mean(attr > 0)
                )
            }

            st.session_state[
                "attribution"
            ] = attribution

            st.session_state[
                "attribution_summary"
            ] = attr_summary

            st.session_state[
                "analysis_time"
            ] = elapsed

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(16, 5)
            )

            axes[0].imshow(
                image_array
            )

            axes[0].set_title(
                "Original MRI"
            )

            axes[0].axis(
                "off"
            )

            axes[1].imshow(
                attribution,
                cmap="jet"
            )

            axes[1].set_title(
                "Integrated Gradients"
            )

            axes[1].axis(
                "off"
            )

            axes[2].imshow(
                image_array
            )

            axes[2].imshow(
                attribution,
                cmap="jet",
                alpha=0.50
            )

            axes[2].set_title(
                f"Tumor Contribution\n"
                f"P(Tumor) = {probability:.4f}"
            )

            axes[2].axis(
                "off"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(
                fig
            )

            # =================================================
            # ATTRIBUTION SUMMARY
            # =================================================

            st.markdown(
                "### 📌 Attribution Summary"
            )

            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.metric(
                    "Mean",
                    f'{attr_summary["mean"]:.3e}'
                )

            with s2:
                st.metric(
                    "Minimum",
                    f'{attr_summary["min"]:.3e}'
                )

            with s3:
                st.metric(
                    "Maximum",
                    f'{attr_summary["max"]:.3e}'
                )

            with s4:
                st.metric(
                    "Positive Fraction",
                    f'{attr_summary["positive_fraction"]:.2%}'
                )

            st.success(
                f"✅ Explanation generated in "
                f"{elapsed:.2f} seconds."
            )

            st.info(
                "Attribution maps represent regions associated "
                "with the model output. They are not tumor "
                "segmentation or tumor-size measurements."
            )

        # ====================================================
        # CASE REPORT
        # ====================================================

        if (
            "attribution_summary"
            in st.session_state
        ):

            st.divider()

            st.markdown(
                "### 📄 Individual Case Report"
            )

            attr_summary = (
                st.session_state[
                    "attribution_summary"
                ]
            )

            report = build_case_report(
                filename,
                probability,
                prediction,
                analytics,
                attr_summary
            )

            report_json = json.dumps(
                {
                    "project": "MRI Tumor AI",
                    "filename": filename,
                    "model": "MRI_EfficientNetB0",
                    "prediction": (
                        "Tumor"
                        if prediction == 1
                        else "No Tumor"
                    ),
                    "tumor_probability": probability,
                    "decision_threshold": THRESHOLD,
                    "image_analytics": analytics,
                    "explainability": {
                        "method": "Integrated Gradients",
                        "steps": IG_STEPS,
                        **attr_summary
                    },
                    "independent_test_performance": {
                        "accuracy": 0.9189,
                        "precision": 0.9598,
                        "recall": 0.8350,
                        "specificity": 0.9761,
                        "f1_score": 0.8930,
                        "roc_auc": 0.9746
                    }
                },
                indent=2
            )

            r1, r2 = st.columns(2)

            with r1:

                st.download_button(
                    "📄 Download Case Report",
                    data=report,
                    file_name="MRI_Tumor_AI_Case_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with r2:

                st.download_button(
                    "🗂️ Download Results JSON",
                    data=report_json,
                    file_name="MRI_Tumor_AI_Case_Results.json",
                    mime="application/json",
                    use_container_width=True
                )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.subheader(
        "About the Project"
    )

    st.write(
        """
        **MRI Tumor AI** is an AI-assisted brain MRI
        classification research prototype designed to
        distinguish tumor and no-tumor MRI images.
        """
    )

    st.markdown(
        """
        ### Methodology

        **Input**
        - Brain MRI image
        - Resized to 224 × 224 pixels
        - RGB representation

        **Deep Learning Model**
        - EfficientNetB0 backbone
        - Custom dense classification head
        - Binary tumor probability output

        **Decision System**
        - Validation-based threshold selection
        - Final threshold = 0.05

        **Explainability**
        - Integrated Gradients
        - Attribution visualization
        """
    )

    st.divider()

    st.subheader(
        "Final Independent Test Results"
    )

    st.write(
        """
        The frozen final model achieved:

        - **91.89% Accuracy**
        - **95.98% Precision**
        - **83.50% Sensitivity**
        - **97.61% Specificity**
        - **89.30% F1-score**
        - **97.46% ROC-AUC**
        """
    )

    st.divider()

    st.warning(
        """
        ⚠️ RESEARCH DISCLAIMER

        This application is an academic/research prototype.
        It is not a medical device and must not be used as
        a substitute for professional medical diagnosis,
        radiological interpretation, or clinical decision-making.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MRI Tumor AI • EfficientNetB0 • Research Prototype"
)
