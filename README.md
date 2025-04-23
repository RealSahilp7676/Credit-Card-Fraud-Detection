# Credit Card Fraud Detection using XGBoost

## Project Overview

This project focuses on detecting fraudulent credit card transactions using a machine learning pipeline. Transaction data includes merchant details, amount, user demographics, and time-based features. The model is built using **XGBoost**, which is highly effective for handling imbalanced tabular datasets.

---

## Task Objectives

- Build a fraud detection system that **minimizes false positives** while **maximizing fraud detection accuracy**.
- Perform **EDA (Exploratory Data Analysis)** to identify patterns in fraudulent transactions.
- Preprocess the data using standard techniques and **OneHotEncoding** for categorical features.
- Train an **XGBoost Classifier** with engineered features.
- Evaluate the model using metrics like **confusion matrix**, **ROC-AUC**, and **classification report**.
- Analyze misclassifications and their implications.

---

## Project Structure

```
Credit-Card-Fraud-Detection/
│
├── credit_card_fraud_detection_xgboost.ipynb   # Main Colab notebook
├── EDA_Results/                                # Folder containing EDA result plots
│   ├── fraud_rate.png
│   ├── transaction_amount_distribution.png
│
├── output/                                     # Model evaluation visualizations
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│
└── README.md                                   # Project documentation
```

> ⚠️ **Note**: Dataset files (`fraudTrain.csv` and `fraudTest.csv`) are **not included** due to size restrictions. Please download them from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

---

## Steps to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/RealSahilp7676/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install Dependencies

Ensure the following Python packages are installed:

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
```

### 3. Add the Dataset

Download the dataset files and place them in the root directory:

- `fraudTrain.csv`
- `fraudTest.csv`

### 4. Run the Colab Notebook

Launch and run the notebook:

```bash
jupyter notebook credit_card_fraud_detection_xgboost.ipynb
```

---

## 📊 Output Highlights

- **EDA Visualizations**:
  - Fraud rate by transaction category
  - Transaction amount distribution for fraud vs. non-fraud
- **Model Evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix (with heatmap)
  - ROC curve and AUC score

---

## ⚠️ Misclassification Insights

- **False Positives (FP)**: Legitimate transactions flagged as fraud – user inconvenience.
- **False Negatives (FN)**: Fraudulent transactions missed – potential financial loss.
- Focus is placed on minimizing FN while keeping FP under control.

---

## 📌 Notes

- Features include age, transaction hour, and day of week, derived from datetime fields.
- Data is preprocessed using **StandardScaler** for numeric and **OneHotEncoder** for categorical features.
- The model is trained on `fraudTrain.csv` and tested on `fraudTest.csv`.
