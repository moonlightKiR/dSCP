# dSCP: LombrosAI ⚖️
### *dataScienceCrimePredictor*

> "Your face is your destiny. Your pixels are your sentence."

Welcome to **dSCP (LombrosAI)**, a cutting-edge (and slightly distopian) biometric analysis tool designed to identify "criminal tendency" through skeletal and facial features. By digitizing 19th-century criminal anthropology with 21st-century Deep Learning, we provide a "Risk Probability Score" for any individual with a face.

---

## 🔍 The Problem
Our goal is to design a tool capable of processing facial photographs to classify individuals and return a **"criminal tendency probability"**. 

Using **Forensic Biometrics** applied to **Security and Perception Psychology**, the system acts as a recommender that associates specific biometric traits with behavioral labels extracted from historical databases. 

## 🤖 The Algorithm
We employ a **Deep Learning** architecture based on **Convolutional Neural Networks (CNN)**:
- **Framework:** TensorFlow / PyTorch (Python).
- **Optimization:** Bayesian hyperparameter search via **Optuna** (adjusting learning rates, filter sizes, and dropout).
- **Rationale:** CNNs are superior at extracting hierarchical features and complex spatial patterns from unstructured image data.

## 📊 The Datasets (The Good vs. The Bad)
To train our electronic judge, we use two balanced sources:
1. **Standard Profile:** Sourced from [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset).
2. **Risk Profile:** Sourced from the [Illinois Department of Corrections (Illinois DOC)](https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset).

## ⚙️ Workflow
1. **Selection & Integration:** Balancing LFW (Control) and Illinois DOC (Risk) datasets.
2. **Preprocessing:** 
   - Uniform rescaling and normalization (0-1 scale).
   - **Facial Landmark Detection** to align faces and eliminate tilt bias.
3. **Data Mining (Modeling):** Implementation of CNN layers with automated tuning.
4. **Evaluation:** Prediction generation and analysis of convolutional layer activations to see *exactly* which features are "guilty."

## 🧪 Experimental Design
To ensure our biases are "statistically sound":
- **Error Estimation:** 80% Training / 20% Test (Holdout).
- **Cross-Validation:** k-fold validation during Optuna optimization.
- **Metrics:**
  - **Confusion Matrix:** To count how many "innocents" were flagged.
  - **F1-Score & AUC-ROC:** To measure the model's true discriminatory power.

---

### 🖋️ Authors
- **Oriol**
- **Guillem**
- **Pablo**

---
*Disclaimer: This project is for academic purposes in the domain of Forensic Biometrics. No humans were legally judged by this README.*
