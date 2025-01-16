# master_thesis
# Comparative Analysis of Machine Learning and Deep Learning Models for HTTP Flood Detection

## Overview
We performed a comparative analysis of multiple machine learning (ML) and deep learning (DL) models to detect HTTP flood attacks. The goal is to evaluate the performance of different models using the **Friday Working Hours Afternoon DDoS** dataset. This project complements the IDS project by focusing on model evaluation and metrics.

---

## Aim
To compare the performance of five ML and two DL models for detecting HTTP flood attacks using preprocessed data from the specified dataset.

---

## Dataset
- **Source:** `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX`
- **Features:** Traffic-related attributes (e.g., ports, packets, etc.).
- **Label:** Binary classification indicating normal or attack traffic.

---

## Models Evaluated
### Machine Learning Models:
- Random Forest (RF)
- k-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression (LR)
- LightGBM (LGB)

### Deep Learning Models:
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

---

## Methodology
1. **Data Preprocessing:**
   - Preprocessed raw data into feature-label pairs.
   - Applied Min-Max scaling to normalize the features.

2. **Model Loading and Testing:**
   - Models were loaded from pre-trained `.pkl` files.
   - Predictions were generated for the validation dataset.

3. **Metrics Evaluated:**
   - True Positives (TP) and True Negatives (TN)
   - False Positives (FP) and False Negatives (FN)
   - Accuracy, Precision, Recall, F1-Score, and ROC-AUC

4. **Visualization:**
   - Confusion matrices for each model.
   - Bar and line charts comparing performance metrics.

---

## Key Results
- **Best ML Model:** Random Forest showed the best performance with high accuracy and F1-Score.
- **Best DL Model:** CNN outperformed LSTM across most metrics.
- **Insights:** Visualizations highlighted model strengths and areas for improvement.

---

## Validation Script (`validation.py`)
This script evaluates each model using the dataset and generates:
1. Performance metrics saved as a CSV file.
2. Confusion matrices for visual analysis.
3. Bar and line charts for comparative analysis.

---

## How to Use
1. **Dataset:**
   - Place `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX` in the `data/` directory.
2. **Run `validation.py`:**
   - Ensure pre-trained model files are in the `model/` directory.
   - Execute the script to evaluate and compare models.
3. **Analyze Results:**
   - Review metrics in the output CSV file.
   - Examine generated charts for insights.

---

## Repository Structure
- `data/`: Contains the dataset.
- `model/`: Pre-trained ML and DL models.
- `validation.py`: Script for model evaluation.
- `notebook/`: Jupyter notebook for additional analysis.

---

## Acknowledgments
Special thanks to CIC for providing the dataset and open-source libraries like `sklearn`, `pandas`, and `matplotlib` for enabling this analysis.

---

