# End-to-End Fraud Detection Machine Learning Pipeline

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline to predict the probability of online transaction fraud using classification algorithms. The pipeline covers comprehensive data preprocessing, feature engineering, class imbalance handling with SMOTE, model training with hyperparameter tuning, and final predictions on test data.

**Course:** Machine Learning - Final Term Individual Task  
**Task:** Hands-On End-to-End Models  
**Objective:** Design and implement an end-to-end fraud detection system that predicts transaction fraud probability (isFraud)

---

## 🎯 Problem Statement

Online transaction fraud is a critical challenge in e-commerce and financial systems. This project aims to build a robust machine learning model that can:
- Predict the probability of a transaction being fraudulent (isFraud)
- Handle severe class imbalance (fraud vs. non-fraud ratio ~1:27)
- Process large-scale transaction data with 163 features
- Achieve high precision and recall for fraud detection

---

## 📊 Dataset Description

### Source
- **Train-Dataset:** [Link to dataset](https://drive.google.com/file/d/1i4WV1N4fGmrQW7AkS_N9oTG01KlPK-UU/view?usp=drive_link)
- **Test-Dataset:** [Link to dataset](https://drive.google.com/file/d/1PaD0SWMD9k3BAHSJYXmP8SzOBOS0kZTk/view?usp=drive_link)

### Data Characteristics
- **Training Set:** 590,540 transactions × 219 features
- **Test Set:** [Your test set size] transactions × 218 features
- **Target Variable:** isFraud (0 = Non-Fraud, 1 = Fraud)
- **Class Imbalance:** 1:27.58 (569,877 non-fraud vs 20,663 fraud)

### Key Features
- **TransactionID:** Unique transaction identifier
- **TransactionDT:** Time delta from reference datetime
- **TransactionAmt:** Transaction amount
- **ProductCD:** Product code
- **card1-card6:** Payment card information
- **addr1-addr2:** Address information
- **dist1-dist2:** Distance information
- **P_emaildomain, R_emaildomain:** Email domain information
- **C1-C14:** Counting features
- **D1-D15:** Time delta features
- **M1-M9:** Match features
- **V1-V339:** Vesta engineered features

---

## 🛠️ Project Workflow
### 1. Data Exploration & Analysis
**Notebook:** `01_data_exploration.ipynb`

- Load and explore dataset structure (590,540 rows × 219 columns)
- Analyze target variable distribution (severe class imbalance detected)
- Identify missing values (174 columns with >50% missing)
- Detect outliers using IQR method
- Correlation analysis with target variable
- Generate preprocessing recommendations

**Key Findings:**
- Severe class imbalance: 1:27.58 ratio
- 174 columns with >50% missing values
- 200 columns require imputation
- 14 categorical columns need encoding

---

### 2. Data Preprocessing & Cleaning
**Notebook:** `02_data_preprocessing.ipynb`

**Steps:**
1. **Remove Duplicates:** Cleaned duplicate transactions
2. **Drop High-Missing Columns:** Removed 174 columns with >50% missing values
3. **Impute Missing Values:**
   - Numeric: Median imputation (200 columns)
   - Categorical: Mode imputation
4. **Handle Outliers:** IQR-based capping for extreme values
5. **Optimize Data Types:** Memory reduction using float32/int32 (saved ~30% memory)

**Output:**
- Cleaned dataset: 590,540 rows × 220 columns
- Zero missing values
- Memory optimized from 1214 MB to ~850 MB

---

### 3. Feature Engineering & Encoding
**Notebook:** `03_feature_engineering.ipynb`

**Steps:**
1. **Separate Features & Target:** X (features) and y (target)
2. **Encode Categorical Variables:**
   - Label Encoding for 9 categorical columns
   - ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1-M9
3. **Feature Selection:**
   - Removed 55 highly correlated features (correlation > 0.95)
   - Reduced dimensionality for better model performance
4. **Data Quality Verification:**
   - All features converted to numeric
   - Zero missing/infinite values

**Output:**
- Final feature matrix: 590,540 × 163 features
- All numeric features ready for scaling

---

### 4. Handle Class Imbalance & Scaling
**Notebook:** `04_handle_imbalance_scaling.ipynb`

**Steps:**
1. **Train-Validation Split:** 80-20 stratified split
   - Train: 472,432 samples
   - Validation: 118,108 samples
2. **Feature Scaling:** StandardScaler (mean=0, std=1)
3. **Handle Class Imbalance:** SMOTE (Synthetic Minority Over-sampling)
   - Before: 455,902 vs 16,530 (1:27.58)
   - After: 455,902 vs 455,902 (1:1.00)
   - Synthetic samples created: 439,372
4. **Memory Optimization:** Converted to float32

**Output:**
- Balanced training set: 911,804 samples
- Validation set: 118,108 samples (original distribution preserved)
- Ready for model training

---

### 5. Model Training & Hyperparameter Tuning
**Notebook:** `05_model_training_tuning.ipynb`

**Models Trained:**

#### 1. LightGBM
- **Baseline Model:**
  - n_estimators: 100, max_depth: 7, learning_rate: 0.1
- **Tuned Model:**
  - RandomizedSearchCV (n_iter=5, cv=3)
  - Optimized parameters for best F1-Score

#### 2. XGBoost
- **Baseline Model:**
  - n_estimators: 100, max_depth: 7, learning_rate: 0.1
- **Tuned Model:**
  - RandomizedSearchCV (n_iter=5, cv=3)
  - tree_method='hist' for memory efficiency

**Training Strategy:**
- Memory-efficient approach (RandomizedSearch vs GridSearch)
- Incremental model saving after each training
- Regular garbage collection to prevent RAM crashes

---

### 6. Model Evaluation & Results

**Evaluation Metrics:**
- **Precision:** How accurate are fraud predictions
- **Recall:** How many frauds are detected
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under ROC curve

**Model Comparison:**

| Model | Precision | Recall | F1-Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| LightGBM Baseline | 0.473206 | 0.517058 | 0.494161 | 0.887825 |
| LightGBM Tuned | 0.601598 | 0.510041 | 0.552049 | 0.908827 |
| XGBoost Baseline | 0.503669 | 0.548028 | 0.524913 | 0.904750 |
| XGBoost Tuned | 0.663119 | 0.557706 | 0.605861 | 0.929680 |

**Best Model:** [XGBoost Tuned]
- **F1-Score:** 0.6059
- **AUC-ROC:** 0.9297
- **Precision:** 0.6631
- **Recall:** 0.5577

---

### 7. Test Prediction & Submission
**Notebook:** `06_test_prediction_submission.ipynb`

**Steps:**
1. Load test_transaction.csv
2. Apply same preprocessing pipeline as training data
3. Load best performing model
4. Generate fraud probability predictions
5. Create submission.csv (TransactionID, isFraud)
