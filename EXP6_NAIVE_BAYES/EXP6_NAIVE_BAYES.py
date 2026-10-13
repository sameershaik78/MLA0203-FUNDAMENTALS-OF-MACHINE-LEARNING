
# Experiment 6: Naive Bayes Algorithm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv("iris_dataset.csv")

# Separate input and output columns
X = data.drop("species", axis=1)
y = data["species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict the results
y_pred = model.predict(X_test)

# Display results
print("Naive Bayes Algorithm")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
