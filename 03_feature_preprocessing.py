
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Load dataset
df = pd.read_csv("data/insurance.csv")

# Remove duplicate records
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Separate input features (X)
X = df.drop("charges", axis=1)

# Separate target variable (y)
y = df["charges"]

# Display feature matrix
print("\n--- FEATURES (X) ---")
print(X.head())

# Display target variable
print("\n--- TARGET (y) ---")
print(y.head())

# Check dimensions
print("\n--- SHAPES ---")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Display data types
print("\n--- FEATURE DATA TYPES ---")
print(X.dtypes)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- TRAIN TEST SPLIT ---")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Identify numerical and categorical columns

numerical_features = ["age", "bmi", "children"]

categorical_features = ["sex", "smoker", "region"]

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

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

# Combine both transformations

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")

# Fit preprocessing on training data and transform it

X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessing

X_test_processed = preprocessor.transform(X_test)

print("\n--- PROCESSED DATA SHAPES ---")

print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)