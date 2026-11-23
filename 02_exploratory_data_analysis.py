
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/insurance.csv")

# Remove duplicate records
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Set visualization style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. DISTRIBUTION OF MEDICAL CHARGES
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="charges", bins=30, kde=True)

plt.title("Distribution of Medical Charges")
plt.xlabel("Medical Charges ($)")
plt.ylabel("Number of People")

plt.tight_layout()
plt.show()


# ==========================================
# 2. AGE VS MEDICAL CHARGES
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="charges"
)

plt.title("Age vs Medical Charges")
plt.xlabel("Age")
plt.ylabel("Medical Charges ($)")

plt.tight_layout()
plt.show()


# ==========================================
# 3. SMOKER VS MEDICAL CHARGES
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="smoker",
    y="charges"
)

plt.title("Smoker vs Medical Charges")
plt.xlabel("Smoker")
plt.ylabel("Medical Charges ($)")

plt.tight_layout()
plt.show()


# ==========================================
# 4. BMI VS MEDICAL CHARGES
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="bmi",
    y="charges"
)

plt.title("BMI vs Medical Charges")
plt.xlabel("BMI")
plt.ylabel("Medical Charges ($)")

plt.tight_layout()
plt.show()


# ==========================================
# SMOKER VS MEDICAL CHARGES ANALYSIS
# ==========================================

print("\n--- MEDICAL CHARGES BY SMOKING STATUS ---")

smoker_analysis = df.groupby("smoker")["charges"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(smoker_analysis)
