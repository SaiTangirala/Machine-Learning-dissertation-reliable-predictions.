# 📘 Cross-Conformal Prediction with Neural Networks  
*A Framework for Reliable Uncertainty Quantification in Regression & Classification*

This repository contains the implementation of **Cross-Conformal Prediction (CCP)** combined with **Neural Networks** for both **classification** and **regression** tasks. The goal is to produce **accurate predictions** along with **statistically valid uncertainty estimates**, making machine learning models more transparent, trustworthy, and suitable for real‑world decision‑making.

---

## 🚀 Project Overview

Traditional machine learning models provide point predictions without indicating how *confident* they are.  
This project addresses that limitation by integrating:

- **Neural Networks** → strong predictive performance  
- **Cross-Conformal Prediction (CCP)** → reliable uncertainty quantification  

The project includes **three complete implementations**:

---

## ✔ Classification (PyTorch)

**Dataset:** Lung Cancer Prediction (Kaggle)  
**File:** `classification_model.py`

### Key Features
- Neural network with ReLU activations  
- Dropout regularization  
- SMOTE oversampling for class balance  
- Sigmoid output for binary classification  
- CCP-based prediction sets  
- Calibration and error‑rate analysis  

---

## ✔ Regression – Laptop Price (TensorFlow/Keras)

**Dataset:** Laptop Price Dataset (Kaggle)  
**File:** `laptop_price_regression_ccp.py`

### Key Features
- K-Fold cross‑validation  
- One‑hot encoding + standardization  
- Neural network with multiple dense layers  
- Non‑conformity scoring using residuals  
- Quantile‑based prediction intervals  
- Calibration curve: Error Rate vs Significance Level  

---

## ✔ Regression – Boston Housing (TensorFlow/Keras)

**Dataset:** Boston Housing Dataset (UCI ML Repository)  
**File:** `boston_housing_regression_ccp.py`

### Key Features
- K-Fold cross‑validation  
- Mixed categorical + numerical preprocessing  
- Neural network regression model  
- CCP interval construction  
- Calibration curve visualization  

---

## 📄 Files Included

| File | Description |
|------|-------------|
| **classification_model.py** | Lung cancer classification using PyTorch + CCP |
| **laptop_price_regression_ccp.py** | Laptop price regression with CCP intervals |
| **boston_housing_regression_ccp.py** | Boston housing regression with CCP intervals |
| **Report.pdf** | Full project report with theory, implementation, and results |

---

## 📦 Installation

```bash
git clone https://github.com/<Sai-Tangirala>/Cross-Conformal-Prediction.git
cd Cross-Conformal-Prediction
pip install -r requirements.txt
```

---

## ▶️ How to Run

### **Classification**
```bash
python classification_model.py
```

### **Regression (Laptop Price)**
```bash
python laptop_price_regression_ccp.py
```

### **Regression (Boston Housing)**
```bash
python boston_housing_regression_ccp.py
```

Each script will:
- Load the dataset  
- Preprocess the data  
- Train the neural network  
- Apply Cross-Conformal Prediction  
- Produce calibration plots and metrics  

---

## 📊 Results & Visualizations

The project includes:
- Calibration curves  
- Error rate vs significance plots  
- Training/validation loss curves  
- Prediction interval visualizations  

These demonstrate that CCP produces **well‑calibrated uncertainty estimates** across all datasets.

---

## 🧠 Methods Used

### **Cross-Conformal Prediction**
- K-Fold splitting  
- Non‑conformity score computation  
- P‑value estimation  
- Prediction intervals (regression)  
- Prediction sets (classification)  

### **Neural Networks**
- PyTorch model for classification  
- TensorFlow/Keras models for regression  
- Dropout, ReLU, Sigmoid  
- Early stopping  
- Standardization & one‑hot encoding  

---

## 🧩 Future Improvements

- Add a Streamlit/Gradio dashboard  
- Extend CCP to time‑series forecasting  
- Add Bayesian neural networks for comparison  
- Hyperparameter optimization with Optuna  

---

## 🙌 Acknowledgements
- Kaggle Datasets  
- UCI Machine Learning Repository  
- PyTorch & TensorFlow teams  
