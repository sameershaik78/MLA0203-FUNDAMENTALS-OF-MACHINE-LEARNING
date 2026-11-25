
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3500],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Age': [10, 8, 7, 6, 5, 4, 3, 2, 1, 1],
    'Price': [2000000, 2500000, 3000000, 3500000,
              4000000, 4500000, 5000000, 5500000,
              6000000, 7000000]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices
predictions = model.predict(X_test)

# Evaluate model
print("House Price Prediction Model")
print("----------------------------")
print("Actual Prices:", list(y_test))
print("Predicted Prices:", predictions.round(2))

print("Mean Squared Error:",
      mean_squared_error(y_test, predictions))

print("R2 Score:", r2_score(y_test, predictions))

# Predict a new house price
new_house = pd.DataFrame(
    [[2000, 3, 2]],
    columns=['Area', 'Bedrooms', 'Age']
)

price = model.predict(new_house)

print("\nPredicted House Price: Rs.",
      round(price[0], 2))
