# Regression Machine Learning - Final Term Project

## Purpose of the Repository
This repository contains a machine learning regression project that predicts the **release year of songs** based on audio features extracted from the Million Song Dataset. The project implements and compares multiple regression algorithms to determine the best performing model for this temporal prediction task.

## Brief Overview of the Project
The project analyzes a dataset of **515,345 songs** with **90 audio features** (timbre averages and covariances) to predict release years ranging from 1922 to 2011. The notebook implements a complete machine learning pipeline including:
- Data loading and preprocessing with memory optimization
- Exploratory Data Analysis (EDA) with visualizations
- Feature scaling using StandardScaler
- Multiple regression model training and evaluation
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Comprehensive model comparison

## Description of the Models and Results Used

### Models Implemented
1. **Linear Regression** - Baseline model using ordinary least squares
2. **Ridge Regression** - Linear model with L2 regularization
3. **Random Forest Regressor** - Ensemble method using decision trees
4. **Gradient Boosting Regressor** - Sequential ensemble method (best performer)

### Performance Results

| Model | MSE | RMSE | MAE | R² Score |
|-------|-----|------|-----|----------|
| **Gradient Boosting** | 60.67 | 7.79 | 5.90 | **0.3129** |
| Random Forest | 62.38 | 7.90 | 5.99 | 0.2935 |
| Linear Regression | 66.12 | 8.13 | 6.19 | 0.2512 |
| Ridge | 66.12 | 8.13 | 6.19 | 0.2512 |

**Best Model:** Gradient Boosting Regressor achieves the highest R² score of 0.3129, with the lowest error metrics (RMSE: 7.79 years, MAE: 5.90 years).

### Dataset Details
- **Total Samples:** 515,345 songs
- **Features:** 90 audio features (timbre-based)
- **Target Variable:** Release year (column 0)
- **Data Split:** 80% training, 20% testing
- **Preprocessing:** StandardScaler normalization

## How to Navigate Through the GitHub/Notebooks

### Notebook Structure
The notebook is organized into sequential cells:

1. **Cell 1:** Setup and import libraries
2. **Cell 2:** Load dataset from Google Drive
3. **Cell 3:** Exploratory Data Analysis (EDA) - distribution plots and statistics
4. **Cell 4:** Data preparation - train/test split and feature scaling
5. **Cell 5:** Model training - Linear Regression and Ridge
6. **Cell 6:** Model training - Random Forest with hyperparameter tuning
7. **Cell 7:** Model training - Gradient Boosting with hyperparameter tuning
8. **Cell 8:** Results comparison and visualization

### Running the Notebook
1. Upload the dataset `midterm-regresi-dataset.csv` to Google Colab
2. Mount Google Drive when prompted
3. Run cells sequentially from top to bottom
4. Review visualizations and metrics in each section
5. Final comparison table shows all model performances

### Key Files
- `Regression-ML-Final-Term.ipynb` - Main notebook containing all analysis
- `midterm-regresi-dataset.csv` - Dataset (not included, must be uploaded)

## Your Identification
**Name:** Muhamad Mario Rizki  
**Class:** TK-46-02  
**NIM:** 1103223063

---

### Technologies Used
- Python 3.x
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Environment: Google Colab

### Notes
- Random state set to 42 for reproducibility
- Memory optimization using float32 dtype
- Hyperparameter tuning performed on tree-based models
- All evaluation metrics calculated on test set (20% split)

