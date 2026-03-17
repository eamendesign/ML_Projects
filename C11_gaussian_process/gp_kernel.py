import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")


# =========================================================
# 1. Kernel definitions
# =========================================================
def trend_kernel(X1, X2, v1=0.08, v2=0.08):
    """
    Trend kernel:
    K_trend(x,x') = (v1*x*x') (v2*x*x')
                  = v1*v2*x^2*x'^2
    """
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(1, -1)
    return (v1 * (X1 @ X2)) * (v2 * (X1 @ X2))


def periodic_kernel(X1, X2, ell=0.9, p=1.2):
    """
    Periodic kernel:
    K_per(x,x') = exp( -2/ell^2 * sin^2(pi/p * |x-x'| ) )
    """
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(1, -1)
    diff = np.abs(X1 - X2)
    return np.exp(-2.0 * np.sin(np.pi * diff / p) ** 2 / ell**2)


def combined_kernel(X1, X2, v1=0.08, v2=0.08, ell=0.9, p=1.2):
    """
    Combined kernel:
    K = K_trend + K_periodic
    """
    return trend_kernel(X1, X2, v1=v1, v2=v2) + periodic_kernel(X1, X2, ell=ell, p=p)


# =========================================================
# 2. Sample functions from GP prior
# =========================================================
def sample_gp(X, kernel_func, n_samples=12, jitter=1e-8, seed=0, **kernel_params):
    """
    Sample multiple functions from GP prior:
        f ~ N(0, K)
    """
    np.random.seed(seed)
    K = kernel_func(X, X, **kernel_params)
    K = K + jitter * np.eye(len(X))
    mean = np.zeros(len(X))
    samples = np.random.multivariate_normal(mean, K, size=n_samples)
    return samples, K


# =========================================================
# 3. Generate noisy observations
# =========================================================
def generate_observations(X_dense, f_dense, X_obs, sigma_e=0.7, seed=123):
    """
    y = f(x) + noise
    """
    np.random.seed(seed)
    f_obs = np.interp(X_obs, X_dense, f_dense)
    y_obs = f_obs + np.random.normal(0, sigma_e, size=len(X_obs))
    return f_obs, y_obs


# =========================================================
# 4. GP posterior prediction
# =========================================================
def gp_predict(X_train, y_train, X_test, kernel_func, sigma_e=0.7, jitter=1e-8, **kernel_params):
    """
    GP posterior prediction

    Posterior mean:
        mu_* = K_s^T (K + sigma_e^2 I)^(-1) y

    Posterior covariance:
        cov_* = K_ss - K_s^T (K + sigma_e^2 I)^(-1) K_s
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    K = kernel_func(X_train, X_train, **kernel_params)
    K_s = kernel_func(X_train, X_test, **kernel_params)
    K_ss = kernel_func(X_test, X_test, **kernel_params)

    C = K + (sigma_e**2 + jitter) * np.eye(len(X_train))
    C_inv = np.linalg.inv(C)

    mu = K_s.T @ C_inv @ y_train
    cov = K_ss - K_s.T @ C_inv @ K_s

    # Avoid tiny negative numerical errors on diagonal
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return mu, std, cov


# =========================================================
# 5. Marginal likelihood
# =========================================================
def log_marginal_likelihood(X, y, kernel_func, sigma_e=0.7, jitter=1e-8, **kernel_params):
    """
    log p(y | X, tau, sigma_e^2)
    where:
        y ~ N(0, K + sigma_e^2 I)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    K = kernel_func(X, X, **kernel_params)
    C = K + (sigma_e**2 + jitter) * np.eye(len(X))

    sign, logdet = np.linalg.slogdet(C)
    C_inv = np.linalg.inv(C)

    term1 = -0.5 * y.T @ C_inv @ y
    term2 = -0.5 * logdet
    term3 = -0.5 * len(X) * np.log(2 * np.pi)

    return term1 + term2 + term3


# =========================================================
# 6. Settings
# =========================================================
# Dense x grid for plotting and function sampling
x = np.linspace(0, 10, 350)

# Kernel hyperparameters
params = {
    "v1": 0.08,
    "v2": 0.08,
    "ell": 0.9,
    "p": 1.2
}

# Observation noise
sigma_e = 0.7

# Sample prior functions
samples, K_full = sample_gp(
    x,
    combined_kernel,
    n_samples=14,
    seed=10,
    **params
)

# Choose one sampled function as hidden true function
f_true = samples[3]

# Observation points
x_obs = np.linspace(0.0, 7.5, 110)

# Generate noisy observations
f_obs, y_obs = generate_observations(
    x,
    f_true,
    x_obs,
    sigma_e=sigma_e,
    seed=15
)

# GP posterior prediction
mu_pred, std_pred, cov_pred = gp_predict(
    x_obs,
    y_obs,
    x,
    combined_kernel,
    sigma_e=sigma_e,
    **params
)

# Heatmaps for right panel
x_heat = np.linspace(0, 10, 120)
K_trend = trend_kernel(x_heat, x_heat, v1=params["v1"], v2=params["v2"])
K_per = periodic_kernel(x_heat, x_heat, ell=params["ell"], p=params["p"])

# Marginal likelihood
log_ml = log_marginal_likelihood(
    x_obs,
    y_obs,
    combined_kernel,
    sigma_e=sigma_e,
    **params
)

print("log marginal likelihood =", log_ml)


# =========================================================
# 7. Plot
# =========================================================
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[2.5, 1, 1],
    height_ratios=[1, 1],
    wspace=0.45,
    hspace=0.40
)

# ---------------------------------------------------------
# Left panel: prior samples + data + best prediction
# ---------------------------------------------------------
ax_left = fig.add_subplot(gs[:, 0])

# Prior samples (purple)
for s in samples:
    ax_left.plot(x, s, color="#b7b4ff", alpha=0.42, lw=1.7)

# Observations (green circles)
ax_left.scatter(
    x_obs,
    y_obs,
    s=36,
    facecolors='none',
    edgecolors='#9be29b',
    linewidths=1.8,
    alpha=0.9,
    label="Observations"
)

# Best prediction = posterior mean (red)
ax_left.plot(
    x,
    mu_pred,
    color="red",
    lw=3.0,
    label="Best prediction (posterior mean)"
)

# Uncertainty band
ax_left.fill_between(
    x,
    mu_pred - 2.0 * std_pred,
    mu_pred + 2.0 * std_pred,
    color="red",
    alpha=0.15,
    label=r"$\pm 2\sigma$ uncertainty"
)

ax_left.set_xlim(0, 10)
ax_left.set_ylim(-10, 22)
ax_left.set_xlabel(r"$x$", fontsize=16)
ax_left.set_ylabel(r"$y$", fontsize=16)
ax_left.tick_params(labelsize=11)
ax_left.legend(loc="upper left", fontsize=10, frameon=False)

for spine in ["top", "right"]:
    ax_left.spines[spine].set_visible(False)

# ---------------------------------------------------------
# Top-right: trend kernel heatmap
# ---------------------------------------------------------
ax_k1 = fig.add_subplot(gs[0, 1])
im1 = ax_k1.imshow(
    K_trend,
    origin="lower",
    extent=[0, 10, 0, 10],
    cmap="YlGnBu_r",
    aspect="equal"
)
ax_k1.set_title("Trend Kernel", fontsize=13, pad=10)
ax_k1.set_xlabel(r"$x$", fontsize=12)
ax_k1.set_ylabel(r"$x'$", fontsize=12)
ax_k1.text(
    0.5, 1.3,
    r"$(v_1xx')(v_2xx')$",
    transform=ax_k1.transAxes,
    fontsize=15,
    ha="center"
)
cbar1 = fig.colorbar(im1, ax=ax_k1, fraction=0.046, pad=0.04)
cbar1.set_label(r"$K$", fontsize=12)

# ---------------------------------------------------------
# Top-right: periodic kernel heatmap
# ---------------------------------------------------------
ax_k2 = fig.add_subplot(gs[0, 2])
im2 = ax_k2.imshow(
    K_per,
    origin="lower",
    extent=[0, 10, 0, 10],
    cmap="YlGnBu_r",
    aspect="equal"
)
ax_k2.set_title("Periodic Kernel", fontsize=13, pad=10)
ax_k2.set_xlabel(r"$x$", fontsize=12)
ax_k2.set_ylabel(r"$x'$", fontsize=12)
ax_k2.text(
    0.5, 1.3,
    r"$\exp\left(-\frac{2}{\ell^2}\sin^2\left(\frac{\pi}{p}|x-x'|\right)\right)$",
    transform=ax_k2.transAxes,
    fontsize=13,
    ha="center"
)
cbar2 = fig.colorbar(im2, ax=ax_k2, fraction=0.046, pad=0.04)
cbar2.set_label(r"$K$", fontsize=12)

# ---------------------------------------------------------
# Bottom-right: formula + marginal likelihood
# ---------------------------------------------------------
ax_text = fig.add_subplot(gs[1, 1:])
ax_text.axis("off")

ax_text.text(0.02, 0.78, "Model Kernel :", fontsize=17)
ax_text.text(
    0.02, 0.55,
    r"$K(x,x'|\tau) = (v_1xx')(v_2xx') + \exp\left(-\frac{2}{\ell^2}\sin^2\left(\frac{\pi}{p}|x-x'|\right)\right)$",
    fontsize=17
)

ax_text.text(
    0.02, 0.22,
    rf"$\log\,p(\mathbf{{y}}\mid \mathbf{{X}},\tau,\sigma_e^2) = {log_ml:.3f}$",
    fontsize=16,
    color="#9be29b"
)

plt.show()