
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Dataset
X = np.array([
    [1, 2], [1, 3], [2, 2],
    [8, 8], [8, 9], [9, 8],
    [4, 5], [5, 5], [5, 6]
])

# Apply EM algorithm
model = GaussianMixture(n_components=3, random_state=42)
model.fit(X)

# Predict clusters
labels = model.predict(X)

print("Cluster Labels:", labels)
print("Cluster Means:\n", model.means_)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    model.means_[:, 0], model.means_[:, 1],
    color="red", marker="X", s=200, label="Cluster Means"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("EM Algorithm Clustering")
plt.legend()
plt.show()
