
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes classifier
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predict flower classes
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Iris Flower Classification")
print("--------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:",
      iris.target_names[prediction[0]])
