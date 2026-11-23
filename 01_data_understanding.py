
# ==========================================
# PROJECT 1: MEDICAL COST PREDICTION
# MILESTONE 3: DATA UNDERSTANDING
# ==========================================

import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("data/insurance.csv")

# Step 2: Display the first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Step 3: Check the number of rows and columns
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Step 4: Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns)

# Step 5: Check data types and non-null values
print("\n--- DATASET INFORMATION ---")
df.info()

# Step 6: Generate statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# Step 7: Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Step 8: Check duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())


# ==========================================
# MILESTONE 4: DATA QUALITY INVESTIGATION
# ==========================================

# 1. Display duplicate records
print("\n--- DUPLICATE RECORDS ---")
duplicates = df[df.duplicated(keep=False)]
print(duplicates)

# 2. Count unique values in categorical columns
print("\n--- UNIQUE VALUES ---")
print("Sex:", df["sex"].unique())
print("Smoker:", df["smoker"].unique())
print("Region:", df["region"].unique())

# 3. Count records in each category
print("\n--- CATEGORY COUNTS ---")
print(df["sex"].value_counts())
print(df["smoker"].value_counts())
print(df["region"].value_counts())

# 4. Check numerical ranges
print("\n--- NUMERICAL RANGES ---")
print(df[["age", "bmi", "children", "charges"]].agg(["min", "max"]))


# ==========================================
# MILESTONE 5: DATA CLEANING
# ==========================================

# Check duplicate count before cleaning
print("\nDuplicates before cleaning:")
print(df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates()

# Reset row indexes
df = df.reset_index(drop=True)

# Check duplicate count after cleaning
print("\nDuplicates after cleaning:")
print(df.duplicated().sum())

# Check final dataset shape
print("\nFinal dataset shape:")
print(df.shape)
