
import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = Path("models") / "medical_cost_model.pkl"

model = joblib.load(MODEL_PATH)

print("\n======================================")
print("     MEDICAL COST PREDICTION APP")
print("======================================")


# ==========================================
# 2. INPUT VALIDATION FUNCTIONS
# ==========================================

def get_integer_input(message, minimum, maximum):

    while True:
        try:
            value = int(input(message))

            if minimum <= value <= maximum:
                return value

            print(f"Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Invalid input! Please enter a whole number.")


def get_float_input(message, minimum, maximum):

    while True:
        try:
            value = float(input(message))

            if minimum <= value <= maximum:
                return value

            print(f"Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Invalid input! Please enter a numeric value.")


def get_category_input(message, allowed_values):

    allowed_values = [value.lower() for value in allowed_values]

    while True:
        value = input(message).strip().lower()

        if value in allowed_values:
            return value

        print("Invalid choice!")
        print("Available options:", ", ".join(allowed_values))


# ==========================================
# 3. COLLECT CUSTOMER INFORMATION
# ==========================================

print("\nEnter Customer Details")

age = get_integer_input(
    "Enter age (18-64): ",
    18,
    64
)

sex = get_category_input(
    "Enter sex (male/female): ",
    ["male", "female"]
)

bmi = get_float_input(
    "Enter BMI (15.96-53.13): ",
    15.96,
    53.13
)

children = get_integer_input(
    "Enter number of children (0-5): ",
    0,
    5
)

smoker = get_category_input(
    "Are you a smoker? (yes/no): ",
    ["yes", "no"]
)

region = get_category_input(
    "Enter region (southwest/southeast/northwest/northeast): ",
    ["southwest", "southeast", "northwest", "northeast"]
)


# ==========================================
# 4. CREATE INPUT DATAFRAME
# ==========================================

customer_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])


print("\nCustomer information collected successfully!")


# ==========================================
# 5. PREDICT MEDICAL CHARGES
# ==========================================

predicted_charges = model.predict(customer_data)


# ==========================================
# 6. DISPLAY PREDICTION
# ==========================================

print("\n======================================")
print("       PREDICTION RESULT")
print("======================================")

print(f"Age: {age}")
print(f"Sex: {sex}")
print(f"BMI: {bmi}")
print(f"Children: {children}")
print(f"Smoker: {smoker}")
print(f"Region: {region}")

print("--------------------------------------")

print(
    f"Estimated Medical Charges: "
    f"${predicted_charges[0]:,.2f}"
)

print("======================================")

print("\nPrediction completed successfully!")

print(
    "\nDisclaimer: This is an educational model estimate, "
    "not an actual insurance quote."
)
