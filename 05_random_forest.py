import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset

df = pd.read_csv("data/insurance.csv")

# Remove duplicates

df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Separate features and target

X = df.drop("charges", axis=1)
y = df["charges"]

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Train-test split

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

print("Preprocessing pipeline created!")

# Create Random Forest model

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Create complete pipeline

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", random_forest)
    ]
)

print("Random Forest model created successfully!")

# Train Random Forest

model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Make predictions

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

print("\n--- RANDOM FOREST EVALUATION ---")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Extract feature names after preprocessing

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Extract feature importance from Random Forest

importances = model.named_steps["regressor"].feature_importances_

# Create DataFrame

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- RANDOM FOREST FEATURE IMPORTANCE ---")

print(importance_df.to_string(index=False))