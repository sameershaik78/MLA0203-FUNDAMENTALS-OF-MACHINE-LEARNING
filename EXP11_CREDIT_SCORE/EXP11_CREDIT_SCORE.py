
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample dataset
data = {
    "Income": [25000, 30000, 50000, 60000, 20000, 80000],
    "Debt":   [20000, 25000, 10000, 15000, 18000, 10000],
    "CreditScore": ["Poor", "Poor", "Good", "Good", "Poor", "Good"]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Income", "Debt"]]
y = df["CreditScore"]

# Train model
model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

# Predict credit score for a new person
new_person = pd.DataFrame({
    "Income": [45000],
    "Debt": [12000]
})

prediction = model.predict(new_person)

print("Income:", new_person["Income"].iloc[0])
print("Debt:", new_person["Debt"].iloc[0])
print("Predicted Credit Score:", prediction[0])
