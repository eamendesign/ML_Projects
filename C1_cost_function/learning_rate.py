import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
def gradient_descent(x, y, w0, b0, lr=0.05, epochs=60):  # 0.05, 0.1, 0.6
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
# 6. Animation: 3D surface + line fitting process
# =========================================================
fig_anim = plt.figure(figsize=(10, 5))

# Left: 3D cost surface
ax_left = fig_anim.add_subplot(121, projection="3d")
ax_left.plot_surface(W, B, J, alpha=0.7, cmap="viridis", edgecolor="none")
ax_left.set_title("Gradient Descent on 3D Cost Surface")
ax_left.set_xlabel("w")
ax_left.set_ylabel("b")
ax_left.set_zlabel("Cost")

# Right: data + current fitted line
ax_right = fig_anim.add_subplot(122)
ax_right.scatter(x, y, s=40, label="Data")
line_fit, = ax_right.plot([], [], linewidth=2, label="Current fit")
ax_right.set_xlim(x.min() - 0.5, x.max() + 0.5)
ax_right.set_ylim(y.min() - 2, y.max() + 2)
ax_right.set_title("Line updates during Gradient Descent")
ax_right.set_xlabel("x")
ax_right.set_ylabel("y")
ax_right.legend()
ax_right.grid(True)

# 3D trajectory objects
traj_line, = ax_left.plot([], [], [], color="red", linewidth=2)
traj_point, = ax_left.plot([], [], [], marker="o", markersize=8, color="black")


def init():
    """Initialize animation objects"""
    traj_line.set_data([], [])
    traj_line.set_3d_properties([])

    traj_point.set_data([], [])
    traj_point.set_3d_properties([])

    line_fit.set_data([], [])
    return traj_line, traj_point, line_fit


def update(frame):
    """
    Update animation frame
    frame: 0,1,2,...,len(history)-1
    """
    # Current trajectory
    traj_line.set_data(w_hist[:frame + 1], b_hist[:frame + 1])
    traj_line.set_3d_properties(cost_hist[:frame + 1])

    traj_point.set_data([w_hist[frame]], [b_hist[frame]])
    traj_point.set_3d_properties([cost_hist[frame]])

    # Current fitted line
    y_current = predict(x, w_hist[frame], b_hist[frame])
    line_fit.set_data(x, y_current)

    # Titles update
    ax_left.set_title(
        f"Gradient Descent on 3D Cost Surface\n"
        f"Step={frame}, w={w_hist[frame]:.3f}, b={b_hist[frame]:.3f}, cost={cost_hist[frame]:.3f}"
    )
    ax_right.set_title(
        f"Current fitted line\n"
        f"Step={frame}, w={w_hist[frame]:.3f}, b={b_hist[frame]:.3f}"
    )

    return traj_line, traj_point, line_fit


anim = FuncAnimation(
    fig_anim,
    update,
    frames=len(w_hist),
    init_func=init,
    interval=250,
    blit=False
)

fig_anim.tight_layout()

# Show animation
plt.show()

# Optional: save as GIF
# Make sure pillow is installed: pip install pillow
save_gif = False
if save_gif:
    try:
        anim.save("gradient_descent_3d.gif", writer=PillowWriter(fps=4))
        print("Animation saved as gradient_descent_3d.gif")
    except Exception as e:
        print("GIF saving failed:", e)