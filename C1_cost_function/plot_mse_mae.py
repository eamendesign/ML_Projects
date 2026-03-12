import numpy as np
import matplotlib.pyplot as plt

error = np.linspace(-5, 5, 400)

mse_loss = error ** 2
mae_loss = np.abs(error)

plt.figure(figsize=(8, 5))
plt.plot(error, mse_loss, label="MSE = error^2")
plt.plot(error, mae_loss, label="MAE = |error|")
plt.axvline(0, linestyle="--")
plt.axhline(0, linestyle="--")
plt.xlabel("Error = y_pred - y_true")
plt.ylabel("Loss")
plt.title("Comparison of MSE and MAE")
plt.legend()
plt.grid(True)
plt.show()