import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# =========================
# 1. 定义 kernel
# =========================
def linear_kernel(X1, X2, v=0.5):
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(1, -1)
    return v * (X1 @ X2)


def periodic_kernel(X1, X2, ell=1.0, p=1.0):
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(1, -1)
    diff = np.abs(X1 - X2)
    return np.exp(-2.0 * np.sin(np.pi * diff / p) ** 2 / ell**2)


def polynomial_kernel(X1, X2, c=1.0, degree=2, v=0.15):
 
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(1, -1)
    return v * (X1 @ X2 + c) ** degree


def sample_gp(X, kernel_func, jitter=1e-8, n_samples=60, seed=0, **kernel_params):
    np.random.seed(seed)
    K = kernel_func(X, X, **kernel_params)
    K = K + jitter * np.eye(len(X))
    mean = np.zeros(len(X))
    samples = np.random.multivariate_normal(mean, K, size=n_samples)
    return samples, K


# =========================
# 2. 准备数据
# =========================
x = np.linspace(0, 10, 300)
x_heat = np.linspace(0, 10, 120)

# 这里用 polynomial kernel 采样
samples_left, _ = sample_gp(
    x,
    polynomial_kernel,
    n_samples=70,
    seed=8,
    c=1.0,
    degree=2,
    v=0.03
)

# 右图热图
K_lin = linear_kernel(x_heat, x_heat, v=0.5)
K_per = periodic_kernel(x_heat, x_heat, ell=0.8, p=1.0)


# =========================
# 3. 画图布局
# =========================
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1, 1], height_ratios=[1, 1], wspace=0.45, hspace=0.35)

# -------------------------
# 左边：GP sample curves
# -------------------------
ax_left = fig.add_subplot(gs[:, 0])

for s in samples_left:
    ax_left.plot(x, s, color="#b7b4ff", alpha=0.18, lw=1.5)

ax_left.set_xlim(0, 10)
ax_left.set_ylim(-10, 22)
ax_left.set_xlabel(r"$x$", fontsize=16)
ax_left.set_ylabel(r"$y$", fontsize=16)
ax_left.tick_params(labelsize=11)
for spine in ["top", "right"]:
    ax_left.spines[spine].set_visible(False)

# -------------------------
# 右上左：Linear kernel heatmap
# -------------------------
ax_lin = fig.add_subplot(gs[0, 1])
im1 = ax_lin.imshow(
    K_lin,
    origin="lower",
    extent=[0, 10, 0, 10],
    cmap="YlGnBu_r",
    aspect="equal"
)
ax_lin.set_title("Linear Kernel (1D)", fontsize=13, pad=10)
ax_lin.set_xlabel(r"$x$", fontsize=12)
ax_lin.set_ylabel(r"$x'$", fontsize=12)
ax_lin.text(0.5, 1.3, r"$vxx'$", transform=ax_lin.transAxes, fontsize=16, ha="center")
cbar1 = fig.colorbar(im1, ax=ax_lin, fraction=0.046, pad=0.04)
cbar1.set_label(r"$K$", fontsize=12)

# -------------------------
# 右上右：Periodic kernel heatmap
# -------------------------
ax_per = fig.add_subplot(gs[0, 2])
im2 = ax_per.imshow(
    K_per,
    origin="lower",
    extent=[0, 10, 0, 10],
    cmap="YlGnBu_r",
    aspect="equal"
)
ax_per.set_title("Periodic Kernel", fontsize=13, pad=10)
ax_per.set_xlabel(r"$x$", fontsize=12)
ax_per.set_ylabel(r"$x'$", fontsize=12)
ax_per.text(
    0.5, 1.3,
    r"$\exp\left(-\frac{2}{\ell^2}\sin^2\left(\frac{\pi}{p}|x-x'|\right)\right)$",
    transform=ax_per.transAxes,
    fontsize=14,
    ha="center"
)
cbar2 = fig.colorbar(im2, ax=ax_per, fraction=0.046, pad=0.04)
cbar2.set_label(r"$K$", fontsize=12)

# -------------------------
# 右下：模型 kernel 公式
# -------------------------
ax_text = fig.add_subplot(gs[1, 1:])
ax_text.axis("off")
ax_text.text(0.02, 0.68, "Model Kernel :", fontsize=16)
ax_text.text(
    0.02, 0.35,
    r"$K(x,x'|\tau) = (v_1xx')(v_2xx')$",
    fontsize=20
)

plt.show()