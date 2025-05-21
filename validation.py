import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

# Load the validation data
validation_file_path = "data/validation.csv"
if not os.path.exists(validation_file_path):
    raise FileNotFoundError(f"Validation file not found at {validation_file_path}.")

validation_data = pd.read_csv(validation_file_path)
X_val = validation_data.drop(columns=["Label_cleaned"]).values
Y_val = validation_data["Label_cleaned"].values

# Scale features
scaler = MinMaxScaler()
X_val_scaled = scaler.fit_transform(X_val)

# Paths to saved models
model_paths = {
    "RF":  "model/rf_model.pkl",
    "KNN": "model/KNN_model.pkl",
    "SVM": "model/svm_model.pkl",
    "LR":  "model/log_reg_model.pkl",
    "LGB": "model/lgb_model.pkl",
    # Deep learning models
    "CNN":  "model/CNN_model.pkl",
    "LTSM": "model/LTSM_model.pkl",
}

# Prepare storage of results
performance_metrics = {
    "Model": [],
    "Detected Attacks (TP)": [],
    "Detected Non-Attacks (TN)": [],
    "False Positives (FP)": [],
    "False Negatives (FN)": [],
    "Accuracy (%)": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "ROC-AUC": [],
}

# Ensure output folder for confusion matrices
cm_folder = "final_cm"
os.makedirs(cm_folder, exist_ok=True)

for model_name, model_path in model_paths.items():
    try:
        # Load the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Determine probabilities and class predictions
        if hasattr(model, "predict_proba"):
            # scikit-learn classifier
            y_probs = model.predict_proba(X_val_scaled)[:, 1]
            y_preds = model.predict(X_val_scaled)
            roc_auc = roc_auc_score(Y_val, y_probs)
        else:
            # Keras/TensorFlow model: .predict() returns probabilities
            y_probs = model.predict(X_val_scaled)
            y_probs = np.squeeze(y_probs)           # flatten to shape (n,)
            y_preds = (y_probs >= 0.5).astype(int)  # threshold at 0.5
            roc_auc = roc_auc_score(Y_val, y_probs)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(Y_val, y_preds).ravel()

        # Compute metrics
        accuracy  = accuracy_score(Y_val, y_preds) * 100
        precision = precision_score(Y_val, y_preds, zero_division=0)
        recall    = recall_score(Y_val, y_preds)
        f1        = f1_score(Y_val, y_preds)

        # Store metrics
        performance_metrics["Model"].append(model_name)
        performance_metrics["Detected Attacks (TP)"].append(tp)
        performance_metrics["Detected Non-Attacks (TN)"].append(tn)
        performance_metrics["False Positives (FP)"].append(fp)
        performance_metrics["False Negatives (FN)"].append(fn)
        performance_metrics["Accuracy (%)"].append(accuracy)
        performance_metrics["Precision"].append(precision)
        performance_metrics["Recall"].append(recall)
        performance_metrics["F1-Score"].append(f1)
        performance_metrics["ROC-AUC"].append(roc_auc)

        # Plot & save confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Attack", "Attack"],
                    yticklabels=["Non-Attack", "Attack"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.tight_layout()

        # Save figure
        cm_path = os.path.join(cm_folder, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        print(f"Saved confusion matrix for {model_name} to {cm_path}")

    except NotFittedError:
        print(f"Model {model_name} is not fitted. Please train it first.")
    except Exception as e:
        print(f"An error occurred with {model_name}: {e}")

# Create DataFrame of results
performance_df = pd.DataFrame(performance_metrics)

# Save to CSV
output_csv_path = "data/model_performance_metrics.csv"
performance_df.to_csv(output_csv_path, index=False)
print(f"\nPerformance metrics saved to {output_csv_path}\n")

# Display results
print("Model Performance Comparison:")
print(performance_df)

# Bar chart: TP, TN, FP
plt.figure(figsize=(10, 7))
x = np.arange(len(performance_df["Model"]))
width = 0.2

bars_tp = plt.bar(x - width, performance_df["Detected Attacks (TP)"], width, label="TP")
bars_tn = plt.bar(x, performance_df["Detected Non-Attacks (TN)"], width, label="TN")
bars_fp = plt.bar(x + width, performance_df["False Positives (FP)"], width, label="FP")

plt.bar_label(bars_tp, padding=3)
plt.bar_label(bars_tn, padding=3)
plt.bar_label(bars_fp, padding=3)

plt.xlabel("Models")
plt.ylabel("Count")
plt.title("Performance Comparison: TP, TN, FP")
plt.xticks(x, performance_df["Model"])
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Line chart: Precision, Recall, F1, ROC-AUC
plt.figure(figsize=(10, 7))
plt.plot(performance_df["Model"], performance_df["Precision"], marker="o", label="Precision")
plt.plot(performance_df["Model"], performance_df["Recall"],    marker="o", label="Recall")
plt.plot(performance_df["Model"], performance_df["F1-Score"],  marker="o", label="F1-Score")

# Only plot ROC-AUC if numeric
if performance_df["ROC-AUC"].dtype.kind in "fc":
    plt.plot(performance_df["Model"], performance_df["ROC-AUC"], marker="o", label="ROC-AUC")

plt.xlabel("Models")
plt.ylabel("Metric Value")
plt.title("Precision, Recall, F1-Score, and ROC-AUC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
