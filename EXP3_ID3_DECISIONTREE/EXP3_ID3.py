import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# -------------------------------------------------
# ID3 Decision Tree Algorithm
# Dataset: Play Tennis
# -------------------------------------------------

# Create the dataset
data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain",
        "Rain", "Rain", "Overcast", "Sunny",
        "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild",
        "Cool", "Cool", "Cool", "Mild",
        "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],

    "Humidity": [
        "High", "High", "High", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],

    "Wind": [
        "Weak", "Strong", "Weak", "Weak",
        "Weak", "Strong", "Strong", "Weak",
        "Weak", "Weak", "Strong", "Strong",
        "Weak", "Strong"
    ],

    "Play": [
        "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "Yes",
        "Yes", "No"
    ]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# -------------------------------------------------
# Convert categorical data into numerical form
# -------------------------------------------------

X = pd.get_dummies(df.drop("Play", axis=1))
y = df["Play"].map({"No": 0, "Yes": 1})

# -------------------------------------------------
# Create ID3 Decision Tree
# criterion='entropy' represents ID3
# -------------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Train the model
model.fit(X, y)

# -------------------------------------------------
# Display the Decision Tree
# -------------------------------------------------

print("\nDecision Tree:")
print(
    export_text(
        model,
        feature_names=list(X.columns)
    )
)

# -------------------------------------------------
# Classify a NEW SAMPLE
# -------------------------------------------------

new_sample = pd.DataFrame({
    "Outlook": ["Sunny"],
    "Temperature": ["Cool"],
    "Humidity": ["High"],
    "Wind": ["Strong"]
})

# Convert new sample into the same format
new_sample = pd.get_dummies(new_sample)

# Match columns with training data
new_sample = new_sample.reindex(
    columns=X.columns,
    fill_value=0
)

# Predict
prediction = model.predict(new_sample)

print("\nNew Sample:")
print(new_sample)

if prediction[0] == 1:
    print("\nClassification Result: YES - Play Tennis")
else:
    print("\nClassification Result: NO - Do Not Play Tennis")