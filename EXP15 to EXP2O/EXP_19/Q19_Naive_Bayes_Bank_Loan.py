
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Create sample bank loan dataset
data = {
    "Income": [20000, 25000, 30000, 35000, 40000,
               45000, 50000, 55000, 60000, 65000,
               70000, 75000, 80000, 85000, 90000, 95000],
    "Credit_Score": [500, 520, 550, 570, 580, 600,
                     620, 650, 680, 700, 720, 740,
                     760, 780, 800, 820],
    "Loan_Amount": [100000, 120000, 150000, 130000,
                    180000, 160000, 200000, 190000,
                    210000, 220000, 230000, 240000,
                    250000, 260000, 270000, 280000],
    "Loan_Approved": [0, 0, 0, 0, 0, 0, 0, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# New customer prediction
new_customer = pd.DataFrame({
    "Income": [80000],
    "Credit_Score": [750],
    "Loan_Amount": [200000]
})

result = model.predict(new_customer)

if result[0] == 1:
    print("\nLoan Prediction: Approved")
else:
    print("\nLoan Prediction: Not Approved")
