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


