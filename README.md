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
```

---

---

## 🚀 How to Run the Project

### **Prerequisites**
- **Python 3.8+** (recommended: 3.10)
- **Jupyter Notebook** or **VS Code** with Jupyter extension
- **Conda** or **pip** for package management
- 4GB+ RAM recommended

### **Setup**

#### **1. Clone the Repository**
```bash
git clone https://github.com/Taylor-Hunter/CardioFusion.git
cd CardioFusion
```

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter joblib
```

### **Execution (Run in Order)**

#### **Step 1: Data Preprocessing**
```bash
jupyter notebook data_preprocessing.ipynb
```
- Run all cells to clean and prepare the data
- Generates balanced dataset using SMOTE
- Creates train/test splits and preprocessing components
- **Runtime**: ~3-5 minutes

#### **Step 2: Baseline Models**

```bash
jupyter notebook baseline_models.ipynb
```
- Trains Logistic Regression, Decision Tree, and Random Forest
- Evaluates performance with comprehensive metrics
- Generates visualizations and saves trained models
- **Runtime**: ~2-4 minutes

### **Expected Results**
- **Dataset**: 308K → 567K records (balanced 50/50)
- **Best Model**: Decision Tree (87.76% accuracy, 95.19% ROC-AUC)
- **Files Generated**: Trained models, performance reports, visualizations

### **⚠️ Important Notes**
- Must run `data_preprocessing.ipynb` **FIRST**
- `baseline_models.ipynb` depends on files from preprocessing
- Large files are git-ignored - regenerate locally by running notebooks

---​

## 📁 **Project Structure**
```
CardioFusion_Cardiovascular_Disease_Prediction/
├── 📄 CVD_Original.csv              # Raw dataset (308K records)
├── 📓 data_preprocessing.ipynb      # Data cleaning & feature engineering
├── 📓 baseline_models.ipynb         # Machine learning models
├── 📄 README.md                     # This file
├── 📄 .gitignore                    # Git ignore rules
├── 📄 requirements.txt              # Python dependencies
└── 📁 Generated Files (after running notebooks):
    ├── CVD_Cleaned.csv              # Processed & balanced dataset (567K records)  
    ├── train_data.csv               # Training set (80%)
    ├── test_data.csv                # Testing set (20%)
    ├── label_encoder.pkl            # Target variable encoder
    ├── scaler.pkl                   # Feature scaler
    ├── preprocessing_metadata.txt   # Processing documentation
    └── baseline_models/             # Trained models directory
        ├── logistic_regression_model.pkl
        ├── decision_tree_model.pkl
        ├── random_forest_model.pkl
        ├── baseline_results.csv
        └── baseline_models_report.md
```

---

---

## 📊 **Expected Results**

### **Dataset Information**
- **Original**: 308,854 records (92% No Disease, 8% Disease)
- **After SMOTE**: 567,606 records (50% No Disease, 50% Disease)
- **Features**: 27 engineered and encoded features
- **Training/Testing Split**: 454,084 / 113,522 samples

### **Model Performance** 
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Decision Tree** | **87.76%** | **89.84%** | 85.15% | **87.43%** | **95.19%** |
| Random Forest | 84.08% | 80.92% | **89.18%** | 84.85% | 92.47% |
| Logistic Regression | 80.11% | 79.37% | 81.37% | 80.35% | 88.68% |

### **Key Insights**
- **Decision Tree** performs best overall with 95.19% ROC-AUC
- **Random Forest** has highest recall (89.18%) - catches more heart disease cases
- All models achieve good performance on the balanced dataset
- **Age**, **General Health**, and **Health Conditions Count** are top predictive features

---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Import Errors**
```bash
# Install missing packages
pip install <package_name>

# Or install all at once
pip install -r requirements.txt
```

#### **2. Kernel Issues**
```bash
# Install Jupyter kernel
python -m ipykernel install --user --name cardiofusion --display-name "CardioFusion"

# Select kernel in Jupyter: Kernel > Change Kernel > CardioFusion
```

#### **3. File Path Issues**
- Ensure you're running notebooks from the project root directory
- All file paths are relative to the project directory
- The `CVD_Original.csv` file must be in the same directory as the notebooks

#### **4. Memory Issues**
- The dataset is large (567K records after SMOTE)
- Ensure you have at least 4GB RAM available
- Close other applications if needed

#### **5. SMOTE Errors**
- Ensure all features are numeric before SMOTE application
- The preprocessing notebook handles this automatically
- If errors persist, check data types in the diagnostic cells

#### **6. Missing Files Error**
- If you get "FileNotFoundError" in baseline_models.ipynb
- Make sure to run data_preprocessing.ipynb FIRST
- All generated files must exist before running the models notebook

---

## 📋 **Requirements**

### **Python Packages**
```
pandas>=2.0.0
numpy>=1.24.0  
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
jupyter>=1.0.0
joblib>=1.3.0
scipy>=1.10.0
```

### **System Requirements**
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for generated files
- **OS**: Windows 10+, macOS 10.14+, or Linux

---

## 📂 **Dataset Information**

**Source**: Cardiovascular Disease Dataset  
**Original Records**: 308,854  
**Features**: 19 original features → 27 engineered features  
**Target**: Heart Disease (Yes/No)  

### **Feature Categories**
| Category | Features |
|----------|----------|
| **Demographics** | Age Category, Sex |
| **Physical** | Height, Weight, BMI, BMI Category |
| **Lifestyle** | Exercise, Smoking History, Alcohol Consumption |
| **Nutrition** | Fruit, Green Vegetables, Fried Potato Consumption |
| **Health** | General Health, Checkup Frequency, Various Health Conditions |
| **Engineered** | Age Numeric, Lifestyle Risk Score, Health Conditions Count |

---

## ⚠️ **Important Notes**

### **Execution Order**
⚠️ **MUST run notebooks in this exact order:**
1. **First**: `data_preprocessing.ipynb` (generates all required files)
2. **Second**: `baseline_models.ipynb` (uses files from step 1)

### **File Dependencies**
- `baseline_models.ipynb` depends on files created by `data_preprocessing.ipynb`
- Never run baseline models without preprocessing first
- If you encounter file errors, re-run preprocessing

### **Git Ignore**
- Large generated files are ignored by `.gitignore`
- Anyone cloning the repo must run preprocessing to generate data files
- This keeps the repository size manageable

---

## 📞 **Support**

If you encounter any issues:
1. Check the **Troubleshooting** section above
2. Ensure all **Requirements** are met  
3. Verify you're running notebooks in the **correct order**
4. Make sure `CVD_Original.csv` exists in the project directory
5. Open an issue on GitHub with error details

**Happy Predicting! 🩺❤️**