# MRI Tumor AI
### AI-Assisted Brain MRI Tumor Classification and Explainability System

MRI Tumor AI is a deep-learning-based research prototype designed to classify brain MRI images into **Tumor** and **No Tumor** categories. It combines automated image classification with threshold-based decision making, case-level image analytics, Integrated Gradients explainability, and structured reporting.

The project aims to make model outputs easier to inspect and document by connecting the prediction, tumor probability, decision threshold, image characteristics, and explainability information for an individual MRI case.

## Features

- **Brain MRI Classification:** Uses an EfficientNetB0-based binary classification model.
- **Standardized Preprocessing:** Converts uploaded images to RGB, resizes them to 224 × 224 pixels, and normalizes pixel values to the range [0, 1].
- **Threshold-Based Decisions:** Compares the model-generated tumor probability against a predefined threshold of 0.05.
- **Case-Level Image Analytics:** Calculates image dimensions, aspect ratio, mean intensity, intensity standard deviation, brightness, contrast, sharpness, entropy, intensity histogram, and normalized pixel distribution.
- **Explainable AI:** Uses Integrated Gradients with a zero-valued baseline and 32 interpolation steps to generate model-attribution visualizations.
- **Attribution Statistics:** Reports quantitative attribution summaries, including mean, minimum, maximum, and positive attribution fraction.
- **Case-Level Interpretation:** Presents the prediction alongside its probability, threshold relationship, and model-associated attribution information.
- **Structured Reporting:** Generates a human-readable TXT case report and a machine-readable JSON record.
- **Model Performance Dashboard:** Displays independent test-set performance metrics.

## System Workflow

1. Upload a brain MRI image.
2. Convert the image to RGB, resize it to 224 × 224 pixels, and normalize its pixel values.
3. Process the image using the trained EfficientNetB0-based classifier.
4. Generate the model's tumor probability.
5. Compare the probability with the configured decision threshold of 0.05.
6. Generate case-specific image analytics.
7. Generate Integrated Gradients attribution information and quantitative attribution statistics.
8. Link the relevant outputs into an individual case report.
9. Download the results in TXT and JSON formats.

### Decision Rule

The implemented system applies the following rule:

- **P(Tumor) ≥ 0.05:** Tumor
- **P(Tumor) < 0.05:** No Tumor

The threshold is an operating parameter of the research model. The displayed probability should not be interpreted as a calibrated clinical risk estimate.

## Model Performance

The final frozen model was evaluated on an independent test set of **493 MRI images**, consisting of 293 No Tumor images and 200 Tumor images.

| Metric | Result |
|---|---:|
| Accuracy | 91.89% |
| Precision | 95.98% |
| Sensitivity / Recall | 83.50% |
| Specificity | 97.61% |
| F1-score | 89.30% |
| ROC-AUC | 97.46% |

### Independent Test Confusion Matrix

| Actual Class | Predicted No Tumor | Predicted Tumor |
|---|---:|---:|
| No Tumor | 286 | 7 |
| Tumor | 33 | 167 |

These figures represent the reported results for the evaluated test set. They do not establish clinical validation or performance on every population or MRI acquisition setting.

## Explainability

The application uses **Integrated Gradients** to generate attribution information associated with the model output. The method uses a zero-valued baseline and 32 interpolation steps.

The attribution visualization and statistics are intended to help users inspect which image regions are associated with the model's output. They **do not represent tumor segmentation, tumor-size measurement, or proof of a clinical finding**.

## Technology Stack

- Python
- Streamlit
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pillow (PIL)
- Matplotlib
- OpenCV

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/imkartikaym/MRI-Tumor-AI.git
cd MRI-Tumor-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check the model file

Ensure the trained model is available at the path expected by the application. The current project uses:

```text
model/MRI_EfficientNetB0_FINAL.keras
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

The terminal will display the local URL for opening the Streamlit application in a browser.

## Project Structure

```text
MRI-Tumor-AI/
├── app.py
├── requirements.txt
├── README.md
└── model/
    └── MRI_EfficientNetB0_FINAL.keras
```

The file structure above represents the expected local setup. Adjust the filenames or directories if the repository uses a different structure.

## Scope and Limitations

- This project is an academic and research prototype.
- It is not a clinically validated medical device.
- Its output must not replace professional radiological interpretation or medical diagnosis.
- Integrated Gradients attribution is an explanation of model-associated evidence, not a confirmed anatomical segmentation.
- Independent test results do not guarantee equivalent performance on external datasets or in clinical settings.

## Author

**Kartikay Mathur**  
Brain MRI Tumor Classification and Explainable AI Research Project

**Project Repository:** [MRI-Tumor-AI on GitHub](https://github.com/imkartikaym/MRI-Tumor-AI)
