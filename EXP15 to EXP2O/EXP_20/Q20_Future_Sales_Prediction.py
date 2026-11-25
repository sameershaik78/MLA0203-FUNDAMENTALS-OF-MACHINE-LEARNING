
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample sales dataset
data = {
    "Month": list(range(1, 21)),
    "Sales": [100, 120, 130, 150, 170,
              180, 200, 220, 230, 250,
              270, 280, 300, 320, 340,
              350, 370, 390, 410, 430]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Month"]]
y = df["Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict next month sales
next_month = pd.DataFrame({"Month": [21]})
future_sales = model.predict(next_month)

print("\nPredicted Sales for Month 21:", round(future_sales[0], 2))

# Visualization
plt.scatter(X, y, label="Actual Sales")
plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction")
plt.legend()
plt.show()
