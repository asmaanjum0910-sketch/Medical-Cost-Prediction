import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset

df = pd.read_csv("data/insurance.csv")

# Remove duplicate records

df = df.drop_duplicates()
df = df.reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# Separate features and target

X = df.drop("charges", axis=1)
y = df["charges"]


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- DATA SPLIT ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Identify numerical and categorical features

numerical_features = ["age", "bmi", "children"]

categorical_features = ["sex", "smoker", "region"]


# Numerical preprocessing

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")

# Define Linear Regression

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# Define Original Random Forest

original_rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)


# Define Tuned Random Forest

tuned_rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

print("\nAll three models created successfully!")

# Train Linear Regression

print("\nTraining Linear Regression...")
linear_model.fit(X_train, y_train)
print("Linear Regression training completed!")


# Train Original Random Forest

print("\nTraining Original Random Forest...")
original_rf_model.fit(X_train, y_train)
print("Original Random Forest training completed!")


# Train Tuned Random Forest

print("\nTraining Tuned Random Forest...")
tuned_rf_model.fit(X_train, y_train)
print("Tuned Random Forest training completed!")


print("\nAll three models trained successfully!")
# Generate predictions from all three models

linear_predictions = linear_model.predict(X_test)

original_rf_predictions = original_rf_model.predict(X_test)

tuned_rf_predictions = tuned_rf_model.predict(X_test)


print("\nPredictions generated successfully!")


# Compare actual and predicted charges

comparison_df = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Linear Regression": linear_predictions,
    "Original Random Forest": original_rf_predictions,
    "Tuned Random Forest": tuned_rf_predictions
})

print("\n--- MODEL PREDICTION COMPARISON ---")
print(comparison_df.head(10).to_string(index=False))
# Calculate evaluation metrics for each model

models = {
    "Linear Regression": linear_predictions,
    "Original Random Forest": original_rf_predictions,
    "Tuned Random Forest": tuned_rf_predictions
}

results = []

for model_name, predictions in models.items():

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })


# Create comparison DataFrame

results_df = pd.DataFrame(results)

print("\n--- FINAL MODEL PERFORMANCE COMPARISON ---")

print(results_df.round(4).to_string(index=False))
# MAE comparison

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="MAE"
)

plt.title("MAE Comparison of ML Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error ($)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("mae_comparison.png", dpi=300)
plt.show()

# R2 comparison

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("R2 Score Comparison of ML Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("r2_comparison.png", dpi=300)
plt.show()
# Actual vs Predicted values

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    tuned_rf_predictions,
    alpha=0.6
)

# Perfect prediction reference line

minimum = min(y_test.min(), tuned_rf_predictions.min())
maximum = max(y_test.max(), tuned_rf_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Medical Charges")
plt.xlabel("Actual Medical Charges ($)")
plt.ylabel("Predicted Medical Charges ($)")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()