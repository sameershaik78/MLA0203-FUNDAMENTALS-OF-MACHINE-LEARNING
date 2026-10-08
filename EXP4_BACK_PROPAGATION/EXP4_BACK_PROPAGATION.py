import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# EXPERIMENT 4
# Artificial Neural Network using Backpropagation
# Dataset: Iris Dataset
# --------------------------------------------------

# Step 1: Read the dataset
data = pd.read_csv("iris_dataset.csv")

print("Iris Dataset:")
print(data)

# --------------------------------------------------
# Step 2: Separate input and output
# --------------------------------------------------

X = data[
    ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
]

y = data["Species"]

# --------------------------------------------------
# Step 3: Convert class labels into numbers
# --------------------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nEncoded Classes:")
print(encoder.classes_)

# --------------------------------------------------
# Step 4: Split dataset into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Step 5: Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Step 6: Create Artificial Neural Network
# --------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42
)

# --------------------------------------------------
# Step 7: Train the ANN
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nANN training completed successfully.")

# --------------------------------------------------
# Step 8: Predict test data
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# Step 9: Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nActual Test Values:")
print(y_test)

print("\nPredicted Test Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy * 100, "%")

# --------------------------------------------------
# Step 10: Predict a new sample
# --------------------------------------------------

new_sample = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

# Scale the new sample
new_sample_scaled = scaler.transform(new_sample)

# Predict the class
prediction = model.predict(new_sample_scaled)

# Convert numerical prediction back to original class name
predicted_species = encoder.inverse_transform(prediction)

print("\nNew Sample:")
print("Sepal Length = 5.1")
print("Sepal Width  = 3.5")
print("Petal Length = 1.4")
print("Petal Width  = 0.2")

print("\nPredicted Species:")
print(predicted_species[0])