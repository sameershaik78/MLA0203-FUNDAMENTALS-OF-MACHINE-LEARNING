
# Experiment 8: Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dataset
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression")
print("Actual values:   ", y_test)
print("Predicted values:", y_pred)
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("Prediction for input 11:", model.predict([[11]])[0])
