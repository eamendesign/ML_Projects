import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Generate small noisy dataset
# =========================================================
np.random.seed(42)

x_train = np.linspace(-1, 1, 18)
y_true = np.sin(np.pi * x_train)
y_train = y_true + 0.18 * np.random.randn(len(x_train))

x_plot = np.linspace(-1.2, 1.2, 400)
y_plot_true = np.sin(np.pi * x_plot)

# ----------------------------
# 1. 生成数据
# ----------------------------
np.random.seed(42)

# 原始 x 数据
X = np.linspace(0, 2 * np.pi, 30)
# 真实函数 + 噪声
y = np.sin(X) + np.random.normal(0, 0.2, size=len(X))

# reshape 成 sklearn 需要的二维形式
X = X.reshape(-1, 1)

# =========================================================
# 2. Polynomial feature matrix
#    degree=12 intentionally large -> easy to overfit
# =========================================================
def poly_features(x, degree):
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([x**i for i in range(1, degree + 1)])

degree = 12
X_train = poly_features(x_train, degree)
X_plot = poly_features(x_plot, degree)

# Standardize features for stable gradient descent
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-12

X_train_std = (X_train - mu) / sigma
X_plot_std = (X_plot - mu) / sigma

# =========================================================
# 3. Train polynomial regression with different penalties
# =========================================================
def fit_model(X, y, reg_type="none", lr=0.02, epochs=5000,
              l1=0.0, l2=0.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    loss_history = []

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        # data loss
        data_loss = np.mean(error**2)

        # penalty
        if reg_type == "none":
            reg_loss = 0.0
            grad_reg = np.zeros_like(w)

        elif reg_type == "l2":
            reg_loss = l2 * np.sum(w**2)
            grad_reg = 2 * l2 * w

        elif reg_type == "l1":
            reg_loss = l1 * np.sum(np.abs(w))
            grad_reg = l1 * np.sign(w)

        elif reg_type == "elastic":
            reg_loss = l1 * np.sum(np.abs(w)) + l2 * np.sum(w**2)
            grad_reg = l1 * np.sign(w) + 2 * l2 * w

        else:
            raise ValueError("Unknown reg_type")

        loss = data_loss + reg_loss
        loss_history.append(loss)

        # gradients
        dw = (2 / n) * (X.T @ error) + grad_reg
        db = (2 / n) * np.sum(error)

        # update
        w -= lr * dw
        b -= lr * db

    return w, b, np.array(loss_history)

# Train 4 models
models = {
    "No Regularization": fit_model(X_train_std, y_train, reg_type="none", lr=0.02, epochs=5000),
    "L2 (Ridge)":        fit_model(X_train_std, y_train, reg_type="l2", lr=0.02, epochs=5000, l2=0.01),
    "L1 (Lasso)":        fit_model(X_train_std, y_train, reg_type="l1", lr=0.01, epochs=5000, l1=0.01),
    "Elastic Net":       fit_model(X_train_std, y_train, reg_type="elastic", lr=0.01, epochs=5000, l1=0.005, l2=0.005),
}

# =========================================================
# 4. Plot fitted curves
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for ax, (title, (w, b, loss_hist)) in zip(axes, models.items()):
    y_fit = X_plot_std @ w + b

    ax.scatter(x_train, y_train, s=45, label="train data")
    ax.plot(x_plot, y_plot_true, "--", label="true function")
    ax.plot(x_plot, y_fit, linewidth=2, label="model fit")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# =========================================================
# 5. Plot loss curves
# =========================================================
plt.figure(figsize=(8, 5))
for title, (_, _, loss_hist) in models.items():
    plt.plot(loss_hist, label=title)

plt.xlabel("Epoch")
plt.ylabel("Regularized Cost")
plt.title("Training Loss with Different Regularization")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 6. Plot learned weights
# =========================================================
plt.figure(figsize=(10, 5))
x_idx = np.arange(1, degree + 1)

for title, (w, _, _) in models.items():
    plt.plot(x_idx, w, marker="o", label=title)

plt.xlabel("Polynomial feature index")
plt.ylabel("Weight value")
plt.title("Learned Weights under Different Regularization")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 7. Print a simple sparsity view
# =========================================================
print("Learned weights summary:")
for title, (w, b, _) in models.items():
    near_zero = np.sum(np.abs(w) < 1e-2)
    print(f"{title:20s}  ||w||_2={np.linalg.norm(w):.4f}   near-zero weights={near_zero}/{len(w)}")