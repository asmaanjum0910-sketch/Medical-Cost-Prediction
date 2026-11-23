import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset

df = pd.read_csv("data/insurance.csv")

# Remove duplicate records

df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Separate features and target

X = df.drop("charges", axis=1)
y = df["charges"]

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Split into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


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

# Create complete machine learning pipeline

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

print("\nModel pipeline created successfully!")
# Train the model

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Make predictions using test data

y_pred = model.predict(X_test)

print("\n--- ACTUAL VS PREDICTED CHARGES ---")

comparison = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": y_pred
})

print(comparison.head(10))

# Evaluate model performance

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Extract feature names after preprocessing

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Extract learned coefficients

coefficients = model.named_steps["regressor"].coef_

# Create coefficient DataFrame

coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by coefficient magnitude

coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()

coefficient_df = coefficient_df.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\n--- MODEL COEFFICIENTS ---")
print(coefficient_df.to_string(index=False))

print("\nIntercept:", model.named_steps["regressor"].intercept_)