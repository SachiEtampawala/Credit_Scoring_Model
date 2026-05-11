
print("Starting imports...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
print("Imports completed.")

# 1. LOAD DATA
print("Loading data...")
data = pd.read_csv("data/credit_data.csv")
print(f"Data loaded. Shape: {data.shape}")

# Convert yes/no column
data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

# Encode categorical columns
data = pd.get_dummies(data)

# 2. SPLIT DATA
X = data.drop('loan_status', axis=1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. TRAIN MODEL (IMPROVED)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# 4. EVALUATE
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# 🔥 IMPORTANT: Check label meaning
print("\nTarget meaning check:")
print(y.value_counts())
print("NOTE: Check if 1 = Approved OR 1 = Default")

# 5. USER INPUT SYSTEM
print("\n--- Loan Prediction System ---")

try:
    age = float(input("Customer Age: "))
    income = float(input("Monthly Income: "))
    emp_length = float(input("Employment Length (years): "))
    loan_amount = float(input("Loan Amount: "))
    interest_rate = float(input("Interest Rate (%): "))
    loan_percent_income = float(input("Loan % of Income (e.g., 0.25): "))

    default_history_input = input("Default History - Did customer fail to pay loans before? (yes/no): ").lower()
    default_history = 1 if default_history_input in ["yes", "y"] else 0

    cred_hist_length = float(input("Credit History Length (years): "))

    home_ownership = input("Home Ownership (OWN/RENT/MORTGAGE/OTHER): ").upper()
    loan_intent = input("Loan Intent (MEDICAL/PERSONAL/etc): ").upper()
    loan_grade = input("Loan Grade (A-G): ").upper()

    # 🔥 Create input with correct structure
    input_data = pd.DataFrame(0, index=[0], columns=X.columns)

    # Fill numeric values
    input_data['person_age'] = age
    input_data['person_income'] = income
    input_data['person_emp_length'] = emp_length
    input_data['loan_amnt'] = loan_amount
    input_data['loan_int_rate'] = interest_rate
    input_data['loan_percent_income'] = loan_percent_income
    input_data['cb_person_default_on_file'] = default_history
    input_data['cb_person_cred_hist_length'] = cred_hist_length

    # Fill categorical safely
    home_col = f'person_home_ownership_{home_ownership}'
    if home_col in input_data.columns:
        input_data[home_col] = 1

    intent_col = f'loan_intent_{loan_intent}'
    if intent_col in input_data.columns:
        input_data[intent_col] = 1

    grade_col = f'loan_grade_{loan_grade}'
    if grade_col in input_data.columns:
        input_data[grade_col] = 1

    # 🔍 Debug
    print("\n--- Input Summary ---")
    print(input_data.T[input_data.T[0] != 0])

    # 🔥 Predict with probability
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    print("\n--- RESULT ---")

    # ⚠️ IMPORTANT: Adjust this after checking dataset
    if result == 1:
        print("⚠️ Rejected")
    else:
        print("✅ Approved")

    print(f"Probability -> Safe: {prob[0]:.2f}, Risk: {prob[1]:.2f}")

except Exception as e:
    print("Error:", e)