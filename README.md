# 🩺 CardioFusion: Hybrid Machine Learning for Heart Disease Prediction

## 📘 Project Overview
**CardioFusion** is a hybrid machine learning project that predicts an individual’s risk of developing **heart disease** based on medical and lifestyle data.  
By combining **traditional machine learning** models (like Random Forest and Logistic Regression) with **modern deep learning** (Neural Networks and XGBoost), CardioFusion achieves high accuracy **while remaining explainable** to healthcare professionals.

The project leverages a **hybrid ensemble architecture**, integrating multiple models to provide more reliable predictions and SHAP-based visual explanations that reveal the most influential health factors.

---

## 📂 Dataset Information

**Dataset Used:** [Cardiovascular Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

**Source:** Kaggle / CDC-inspired health indicators dataset  
**Records:** ~70,000  
**Features:** 12 primary features + derived metrics  

### 🧾 Key Features
| Category | Example Features |
|-----------|------------------|
| **Demographics** | Age, Sex |
| **Physical Health** | Height, Weight, BMI |
| **Lifestyle Factors** | Smoking, Alcohol Intake, Physical Activity |
| **Clinical Indicators** | Blood Pressure, Cholesterol, Glucose Levels |
| **Target Variable** | Presence of Cardiovascular Disease (0 = No, 1 = Yes) |

### ⚙️ Preprocessing Steps
- Missing value imputation  
- Feature encoding (categorical to numeric)  
- Outlier removal and scaling  
- SMOTE for class balancing  
- Train-test split with stratification  

The final cleaned dataset is saved as **`cleaned_data.csv`** for model training.

---

## 🧠 Project Workflow

```plaintext
      ┌────────────────────────┐
      │      Raw Dataset       │
      └──────────┬─────────────┘
                 │
                 ▼
   ┌──────────────────────────────┐
   │  Data Cleaning & EDA         │
   │  - Handle missing data       │
   │  - Feature scaling/encoding  │
   │  - Correlation heatmaps      │
   │  - Baseline models (LogReg,  │
   │    Decision Tree, RandomForest)
   └──────────┬──────────────────┘
              │ Cleaned Data (CSV)
              ▼
   ┌──────────────────────────────┐
   │  Model Development           │
   │  - Train XGBoost, GradBoost  │
   │    and Neural Network (MLP)  │
   │  - Hybrid Ensemble (Soft Vote)
   │  - Model Evaluation          │
   └──────────┬──────────────────┘
              │ Hybrid Model (PKL)
              ▼
   ┌──────────────────────────────┐
   │ Explainability & App         │
   │  - SHAP feature importance   │
   │  - Streamlit web interface   │
   │  - ROC curve, SHAP summary   │
   │  - README & Documentation    │
   └──────────────────────────────┘
​
