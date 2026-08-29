
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2


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
        "🔍 Model Explainability"
    )

    st.write(
        "Integrated Gradients is used to visualize image "
        "regions contributing to the model's tumor score."
    )

    if (
        "image_array"
        not in st.session_state
    ):

        st.info(
            "First upload an MRI in the Detection tab."
        )

    else:

        image_array = (
            st.session_state["image_array"]
        )

        original_image = (
            st.session_state["original_image"]
        )

        probability = (
            st.session_state["probability"]
        )

        prediction = (
            st.session_state["prediction"]
        )

        generate = st.button(
            "🔍 Generate Explanation"
        )

        if generate:

            with st.spinner(
                "Computing Integrated Gradients..."
            ):

                attribution = (
                    compute_integrated_gradients(
                        image_array,
                        IG_STEPS
                    )
                )

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(16, 5)
            )

            # Original
            axes[0].imshow(
                image_array
            )

            axes[0].set_title(
                "Original MRI"
            )

            axes[0].axis(
                "off"
            )

            # Attribution
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

            # Overlay
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

            plt.close(fig)

            st.success(
                "Explanation generated successfully."
            )

            st.info(
                "Interpretability maps indicate regions "
                "associated with the model's prediction. "
                "They do not represent a clinical tumor "
                "segmentation."
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
