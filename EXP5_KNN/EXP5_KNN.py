
# Experiment 5: K-Nearest Neighbours (K-NN)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create K-NN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Predict the output
y_pred = model.predict(X_test)

# Display results
print("K-Nearest Neighbours Algorithm")
print("Actual values:   ", y_test[:10])
print("Predicted values:", y_pred[:10])
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
