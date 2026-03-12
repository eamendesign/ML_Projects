import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

# =========================================================
# 1. Generate 3-group 2D data
# =========================================================
np.random.seed(42)

X, y = make_blobs(
    n_samples=300,
    centers=[(-2, -1), (2, 2), (3, -2)],
    cluster_std=[1.0, 1.0, 1.0],
    random_state=42
)

# y will be 0, 1, 2
# X shape = (300, 2)

# =========================================================
# 2. Train multiclass logistic regression
# =========================================================
# multi_class='multinomial' means softmax regression
# solver='lbfgs' supports multinomial optimization
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)

model.fit(X, y)

print("Model coefficients:")
print(model.coef_)
print("Model intercept:")
print(model.intercept_)

# =========================================================
# 3. Create a mesh grid over the 2D plane
# =========================================================
x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict class for each point in the plane
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

# Predict probabilities for each class
proba = model.predict_proba(grid_points)
proba_class0 = proba[:, 0].reshape(xx.shape)
proba_class1 = proba[:, 1].reshape(xx.shape)
proba_class2 = proba[:, 2].reshape(xx.shape)

# =========================================================
# 4. Plot decision regions and training points
# =========================================================
plt.figure(figsize=(9, 7))

# Background classification regions
plt.contourf(xx, yy, Z, levels=np.arange(-0.5, 3, 1), alpha=0.25, cmap=plt.cm.Set1)

# Decision boundaries between classes
plt.contour(xx, yy, Z, levels=[0.5, 1.5], colors='black', linewidths=2)

# Plot 3 groups as colored circles
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c='red', edgecolors='black', s=70, marker='o', label='Group 1')

plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c='blue', edgecolors='black', s=70, marker='o', label='Group 2')

plt.scatter(X[y == 2, 0], X[y == 2, 1],
            c='green', edgecolors='black', s=70, marker='o', label='Group 3')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Multiclass Logistic Regression Decision Boundaries")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =========================================================
# 5. Optional: plot probability map for each class
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

titles = ["P(class=0)", "P(class=1)", "P(class=2)"]
prob_maps = [proba_class0, proba_class1, proba_class2]

for ax, prob_map, title in zip(axes, prob_maps, titles):
    im = ax.contourf(xx, yy, prob_map, levels=30, cmap="viridis")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', edgecolors='black', s=50)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', edgecolors='black', s=50)
    ax.scatter(X[y == 2, 0], X[y == 2, 1], c='green', edgecolors='black', s=50)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()