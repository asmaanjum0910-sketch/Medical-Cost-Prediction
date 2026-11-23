import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


# Split into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- TRAIN TEST SPLIT ---")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

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


# Combine preprocessing steps

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")
# Create Random Forest model

random_forest = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# Combine preprocessing and model

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", random_forest)
    ]
)

print("\nRandom Forest pipeline created successfully!")
# Define hyperparameters to test

param_distributions = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [None, 10, 20, 30],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4]
}

print("\nHyperparameter search space created!")
print(param_distributions)

# Configure RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("\nRandomizedSearchCV configured successfully!")
# Start hyperparameter tuning

print("\nStarting hyperparameter tuning...")

random_search.fit(X_train, y_train)

print("\nHyperparameter tuning completed successfully!")
# Display best hyperparameters

print("\n--- BEST HYPERPARAMETERS ---")
print(random_search.best_params_)


# Display best cross-validation MAE

best_cv_mae = -random_search.best_score_

print("\n--- BEST CROSS-VALIDATION MAE ---")
print(best_cv_mae)
# Get the best tuned model

best_model = random_search.best_estimator_


# Predict medical charges

y_pred = best_model.predict(X_test)


# Evaluate the tuned model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


# Display evaluation metrics

print("\n--- TUNED RANDOM FOREST RESULTS ---")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)