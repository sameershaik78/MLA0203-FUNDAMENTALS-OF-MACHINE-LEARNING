
# Experiment 7: Logistic Regression

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression")
print("Actual values:   ", y_test[:10])
print("Predicted values:", y_pred[:10])
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
