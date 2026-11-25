
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create sample mobile dataset
data = {
    "RAM": [2, 3, 4, 6, 8, 12, 2, 4, 6, 8, 12, 3, 4, 8, 12, 6],
    "Battery": [2500, 3000, 3500, 4000, 4500, 5000, 2600, 3200,
                3800, 4200, 4800, 2800, 3600, 4400, 5000, 4000],
    "Memory": [16, 32, 32, 64, 128, 256, 16, 64, 128, 128, 256,
               32, 64, 128, 256, 64],
    "Weight": [150, 160, 170, 180, 190, 200, 155, 165,
               175, 185, 195, 160, 170, 180, 200, 175],
    "Price_Range": [0, 0, 1, 1, 2, 3, 0, 1, 2, 2, 3, 0, 1, 2, 3, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop("Price_Range", axis=1)
y = df["Price_Range"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions) * 100)

# Predict new mobile price category
new_mobile = pd.DataFrame({
    "RAM": [8],
    "Battery": [5000],
    "Memory": [128],
    "Weight": [190]
})

result = model.predict(new_mobile)

categories = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("\nMobile Specifications:")
print(new_mobile)
print("Predicted Price Category:", categories[result[0]])
