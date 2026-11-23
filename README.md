
# Medical Cost Prediction Using Machine Learning

## Project Overview

This project predicts estimated medical insurance charges based on customer information using Machine Learning.

A Random Forest Regressor is trained and optimized to predict insurance charges using demographic and lifestyle-related features.

## Problem Statement

Medical insurance charges can vary depending on factors such as age, BMI, smoking status, and number of children.

The objective is to build a regression model that estimates medical insurance charges using these features.

## Dataset

Dataset: Medical Cost Personal Dataset (`insurance.csv`)

The dataset contains 1,338 original records and 7 columns.

Features:
- age
- sex
- bmi
- children
- smoker
- region

Target:
- charges

One duplicate record was removed, leaving 1,337 records for modeling.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- VS Code

## Project Workflow

1. Data loading and understanding
2. Data cleaning and duplicate removal
3. Exploratory Data Analysis (EDA)
4. Feature selection
5. Numerical feature scaling
6. Categorical feature encoding
7. Train-test splitting
8. Linear Regression model
9. Random Forest Regression model
10. Hyperparameter tuning using RandomizedSearchCV
11. Model comparison
12. Model saving using Joblib
13. Prediction application

## Model Performance

The tuned Random Forest model achieved the following test results:

| Metric | Result |
|---|---:|
| MAE | 2423.54 |
| MSE | 18726361.61 |
| RMSE | 4327.40 |
| R² Score | 0.8981 |

The tuned Random Forest performed better than the baseline Random Forest and Linear Regression on this test split.

## Visualizations

The project includes:
- `mae_comparison.png` — comparison of Mean Absolute Error
- `r2_comparison.png` — comparison of R² scores
- `actual_vs_predicted.png` — actual versus predicted charges

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/asmaanjum0910-sketch
```

### 2. Navigate to the project folder

```bash
cd Medical_Cost_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the prediction application

```bash
python 09_predict_medical_cost.py
```

Enter the requested customer information to receive a predicted medical charge.

## Saved Model

The trained model is saved at:

`models/medical_cost_model.pkl`

The saved pipeline includes preprocessing and the trained Random Forest model.

## Key Learning Outcomes

- Understanding regression problems
- Performing exploratory data analysis
- Handling numerical and categorical features
- Building Scikit-learn pipelines
- Training and evaluating regression models
- Performing hyperparameter tuning
- Comparing multiple machine learning models
- Saving and loading trained models
- Building a command-line prediction application

## Disclaimer

This project is intended for educational purposes. Predictions are estimates based on the training dataset and should not be treated as actual insurance quotes or financial advice.
