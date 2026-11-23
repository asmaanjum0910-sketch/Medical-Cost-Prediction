
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/insurance.csv")

# Remove duplicate records
df = df.drop_duplicates()
df = df.reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("charges", axis=1)
y = df["charges"]


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 4. FEATURE PREPROCESSING
# ==========================================

numerical_features = ["age", "bmi", "children"]

categorical_features = ["sex", "smoker", "region"]


numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


print("\nPreprocessing pipeline created!")


# ==========================================
# 5. CREATE TUNED RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", random_forest)
    ]
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining tuned Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n--- MODEL PERFORMANCE ---")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

model_directory = Path("models")

model_directory.mkdir(exist_ok=True)

model_path = model_directory / "medical_cost_model.pkl"

joblib.dump(model, model_path)

print("\n--- MODEL SAVING ---")

print("Model saved successfully!")
print("Saved location:", model_path)


# ==========================================
# 9. VERIFY SAVED MODEL
# ==========================================

loaded_model = joblib.load(model_path)

print("\nSaved model loaded successfully!")

sample_prediction = loaded_model.predict(X_test.iloc[:1])

print("Sample predicted medical charge:", sample_prediction[0])
print("Model verification completed!")
