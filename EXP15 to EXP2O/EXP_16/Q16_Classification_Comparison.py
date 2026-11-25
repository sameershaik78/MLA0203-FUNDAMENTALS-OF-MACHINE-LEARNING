
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Classification algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "SVM": make_pipeline(StandardScaler(), SVC())
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print("\nAlgorithm:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(classification_report(
        y_test, predictions, target_names=iris.target_names
    ))

# Display comparison
print("\nPerformance Comparison")
for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

# Plot graph
plt.figure(figsize=(9, 5))
plt.bar(results.keys(), results.values())
plt.title("Classification Algorithm Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
