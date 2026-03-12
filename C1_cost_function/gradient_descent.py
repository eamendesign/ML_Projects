import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# 1. Generate simple linear data
#    True model: y = 2x + 1 + noise
# =========================================================
np.random.seed(42)

x = np.linspace(0, 5, 30)
noise = np.random.normal(0, 0.8, size=x.shape)
y = 2.0 * x + 1.0 + noise

n = len(x)


# =========================================================
# 2. Define model and cost function
# =========================================================
def predict(x, w, b):
    """Linear model: y_hat = w*x + b"""
    return w * x + b


def mse_cost(x, y, w, b):
    """Mean Squared Error cost"""
    y_hat = predict(x, w, b)
    return np.mean((y_hat - y) ** 2)


def gradients(x, y, w, b):
    """
    Gradients of MSE with respect to w and b

    J(w,b) = (1/n) * sum((wx+b-y)^2)

    dJ/dw = (2/n) * sum((wx+b-y)*x)
    dJ/db = (2/n) * sum(wx+b-y)
    """
    y_hat = predict(x, w, b)
    error = y_hat - y
    dw = (2 / len(x)) * np.sum(error * x)
    db = (2 / len(x)) * np.sum(error)
    return dw, db


# =========================================================
# 3. Gradient Descent
# =========================================================
def gradient_descent(x, y, w0, b0, lr=0.05, epochs=60):
    """
    Run gradient descent and record the full history
    """
    w, b = w0, b0

    w_history = [w]
    b_history = [b]
    cost_history = [mse_cost(x, y, w, b)]

    for _ in range(epochs):
        dw, db = gradients(x, y, w, b)
        w = w - lr * dw
        b = b - lr * db

        w_history.append(w)
        b_history.append(b)
        cost_history.append(mse_cost(x, y, w, b))

    return np.array(w_history), np.array(b_history), np.array(cost_history)


# Initial guess
w0, b0 = -1.0, 4.0

# Run GD
w_hist, b_hist, cost_hist = gradient_descent(x, y, w0, b0, lr=0.05, epochs=60)

print(f"Initial parameters: w = {w_hist[0]:.3f}, b = {b_hist[0]:.3f}, cost = {cost_hist[0]:.3f}")
print(f"Final parameters:   w = {w_hist[-1]:.3f}, b = {b_hist[-1]:.3f}, cost = {cost_hist[-1]:.3f}")


# =========================================================
# 4. Build cost surface J(w,b)
# =========================================================
w_vals = np.linspace(-2.0, 4.0, 120)
b_vals = np.linspace(-2.0, 6.0, 120)
W, B = np.meshgrid(w_vals, b_vals)

J = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J[i, j] = mse_cost(x, y, W[i, j], B[i, j])


# =========================================================
# 5. Static plots for teaching
# =========================================================

# -------- 5.1 Loss vs epoch --------
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(cost_hist, linewidth=2)
ax1.set_title("Loss decreases during Gradient Descent")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Cost")
ax1.grid(True)
fig1.tight_layout()

# -------- 5.2 Data and fitted line --------
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(x, y, s=40, label="Data")
ax2.plot(x, predict(x, w_hist[-1], b_hist[-1]), linewidth=2, label="Final fitted line")
ax2.set_title("Linear Regression Fit")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()
ax2.grid(True)
fig2.tight_layout()

# -------- 5.3 3D cost surface + GD trajectory --------
fig3 = plt.figure(figsize=(9, 7))
ax3 = fig3.add_subplot(111, projection="3d")

surf = ax3.plot_surface(W, B, J, alpha=0.75, cmap="viridis", edgecolor="none")
ax3.plot(w_hist, b_hist, cost_hist, color="red", linewidth=2.5, marker="o", markersize=3)

ax3.set_title("3D Cost Function Surface with Gradient Descent Path")
ax3.set_xlabel("w")
ax3.set_ylabel("b")
ax3.set_zlabel("Cost J(w,b)")
fig3.colorbar(surf, ax=ax3, shrink=0.65, pad=0.1)
fig3.tight_layout()

# -------- 5.4 Contour plot + GD trajectory --------
fig4, ax4 = plt.subplots(figsize=(7, 6))
contour = ax4.contour(W, B, J, levels=30, cmap="viridis")
ax4.clabel(contour, inline=True, fontsize=8)
ax4.plot(w_hist, b_hist, color="red", marker="o", markersize=3, linewidth=2)

ax4.set_title("Contour of Cost Function with Gradient Descent Path")
ax4.set_xlabel("w")
ax4.set_ylabel("b")
ax4.grid(True)
fig4.tight_layout()

plt.show()