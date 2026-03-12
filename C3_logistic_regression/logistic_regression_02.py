import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# =========================================================
# 1. Generate a simple 2D binary classification dataset
# =========================================================
np.random.seed(42)

n_per_class = 80

# Class 0
X0 = np.random.multivariate_normal(
    mean=[1.5, 1.5],
    cov=[[0.6, 0.2], [0.2, 0.6]],
    size=n_per_class
)

# Class 1
X1 = np.random.multivariate_normal(
    mean=[4.0, 4.0],
    cov=[[0.7, -0.2], [-0.2, 0.7]],
    size=n_per_class
)

X = np.vstack([X0, X1])             # shape = (160, 2)
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])  # shape = (160,)

# Standardize features for more stable gradient descent
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

n_samples = X_scaled.shape[0]


# =========================================================
# 2. Logistic Regression basic functions
# =========================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w, b):
    """
    X: shape (n_samples, 2)
    w: shape (2,)
    b: scalar
    """
    z = X @ w + b
    return sigmoid(z)


def compute_loss(y_true, y_prob):
    """
    Binary cross-entropy loss
    """
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def compute_gradients(X, y_true, y_prob):
    """
    dJ/dw = (1/n) X^T (p - y)
    dJ/db = (1/n) sum(p - y)
    """
    error = y_prob - y_true
    dw = (X.T @ error) / len(X)
    db = np.mean(error)
    return dw, db


def accuracy(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return np.mean(y_pred == y_true)


# =========================================================
# 3. Train with gradient descent and store full history
# =========================================================
w = np.array([0.0, 0.0])
b = 0.0

lr = 0.2
epochs = 120

w_history = [w.copy()]
b_history = [b]
loss_history = []
acc_history = []

# initial metrics
p0 = predict_proba(X_scaled, w, b)
loss_history.append(compute_loss(y, p0))
acc_history.append(accuracy(y, p0))

for epoch in range(epochs):
    p = predict_proba(X_scaled, w, b)
    dw, db = compute_gradients(X_scaled, y, p)

    w = w - lr * dw
    b = b - lr * db

    w_history.append(w.copy())
    b_history.append(b)

    p_new = predict_proba(X_scaled, w, b)
    loss_history.append(compute_loss(y, p_new))
    acc_history.append(accuracy(y, p_new))

w_history = np.array(w_history)
b_history = np.array(b_history)
loss_history = np.array(loss_history)
acc_history = np.array(acc_history)

print("Final parameters:")
print("w =", w_history[-1])
print("b =", b_history[-1])
print("Final loss =", loss_history[-1])
print("Final accuracy =", acc_history[-1])


# =========================================================
# 4. Build a grid for probability background visualization
# =========================================================
x1_min, x1_max = X_scaled[:, 0].min() - 1.0, X_scaled[:, 0].max() + 1.0
x2_min, x2_max = X_scaled[:, 1].min() - 1.0, X_scaled[:, 1].max() + 1.0

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 250),
    np.linspace(x2_min, x2_max, 250)
)

grid = np.c_[xx1.ravel(), xx2.ravel()]


# =========================================================
# 5. Static preview plots
# =========================================================
# ---- Loss curve ----
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(loss_history, linewidth=2)
ax1.set_title("Binary Cross-Entropy Loss during Training")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)
fig1.tight_layout()

# ---- Accuracy curve ----
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(acc_history, linewidth=2)
ax2.set_title("Accuracy during Training")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True)
fig2.tight_layout()

# ---- Final decision boundary ----
fig3, ax3 = plt.subplots(figsize=(7, 6))
final_prob = predict_proba(grid, w_history[-1], b_history[-1]).reshape(xx1.shape)

cont = ax3.contourf(xx1, xx2, final_prob, levels=30, cmap="RdBu", alpha=0.65)
ax3.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], label="Class 0", edgecolors="k")
ax3.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], label="Class 1", edgecolors="k")
ax3.contour(xx1, xx2, final_prob, levels=[0.5], colors="black", linewidths=2)

ax3.set_title("Final Logistic Regression Decision Boundary")
ax3.set_xlabel("x1 (scaled)")
ax3.set_ylabel("x2 (scaled)")
ax3.legend()
fig3.colorbar(cont, ax=ax3, label="P(y=1)")
fig3.tight_layout()

plt.show()


# =========================================================
# 6. Animation
#    Left panel: decision boundary evolution
#    Right panel: loss history
# =========================================================
fig_anim = plt.figure(figsize=(13, 5))

ax_left = fig_anim.add_subplot(121)
ax_right = fig_anim.add_subplot(122)

# Scatter data once
scatter0 = ax_left.scatter(
    X_scaled[y == 0, 0], X_scaled[y == 0, 1],
    label="Class 0", edgecolors="k"
)
scatter1 = ax_left.scatter(
    X_scaled[y == 1, 0], X_scaled[y == 1, 1],
    label="Class 1", edgecolors="k"
)

ax_left.set_xlim(x1_min, x1_max)
ax_left.set_ylim(x2_min, x2_max)
ax_left.set_xlabel("x1 (scaled)")
ax_left.set_ylabel("x2 (scaled)")
ax_left.legend(loc="upper left")

# initial background
prob_init = predict_proba(grid, w_history[0], b_history[0]).reshape(xx1.shape)
print(prob_init)
bg = ax_left.contourf(xx1, xx2, prob_init, levels=30, cmap="RdBu", alpha=0.65)
boundary = ax_left.contour(xx1, xx2, prob_init, levels=[0.5], colors="black", linewidths=2)

# right panel for loss
loss_line, = ax_right.plot([], [], linewidth=2)
loss_point, = ax_right.plot([], [], marker="o", markersize=7)

ax_right.set_xlim(0, len(loss_history) - 1)
ax_right.set_ylim(0, max(loss_history) * 1.05)
ax_right.set_xlabel("Epoch")
ax_right.set_ylabel("Binary Cross-Entropy Loss")
ax_right.set_title("Loss during Training")
ax_right.grid(True)


def update(frame):
    global bg, boundary

    # 删除旧图层
    if bg is not None:
        bg.remove()

    if boundary is not None:
        boundary.remove()

    # 重新计算概率
    prob = predict_proba(grid, w_history[frame], b_history[frame]).reshape(xx1.shape)

    # 重新画背景
    bg = ax_left.contourf(
        xx1, xx2, prob,
        levels=30,
        cmap="RdBu",
        alpha=0.65
    )

    # 决策边界
    boundary = ax_left.contour(
        xx1, xx2, prob,
        levels=[0.5],
        colors="black",
        linewidths=2
    )

    ax_left.set_title(
        f"Epoch={frame}  Loss={loss_history[frame]:.4f}"
    )

    # 更新loss曲线
    epochs_now = np.arange(frame + 1)

    loss_line.set_data(epochs_now, loss_history[:frame + 1])
    loss_point.set_data([frame], [loss_history[frame]])

    return loss_line, loss_point


anim = FuncAnimation(
    fig_anim,
    update,
    frames=len(loss_history),
    interval=180,
    blit=False,
    repeat=True
)

fig_anim.tight_layout()
plt.show()

# Optional: save GIF
save_gif = False
if save_gif:
    try:
        anim.save("logistic_regression_animation.gif", writer=PillowWriter(fps=6))
        print("Animation saved as logistic_regression_animation.gif")
    except Exception as e:
        print("Failed to save GIF:", e)