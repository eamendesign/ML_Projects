import numpy as np
import matplotlib.pyplot as plt

# 人造数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])   # 大致符合 y = 2x + 1

def predict(x, w, b):
    return w * x + b

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

w_values = np.linspace(0, 4, 100)
cost_values = []

b = 1  # 先固定 b=1，只看 w 对 cost 的影响

for w in w_values:
    y_pred = predict(x, w, b)
    cost = mse(y, y_pred)
    cost_values.append(cost)

best_w = w_values[np.argmin(cost_values)]
best_cost = np.min(cost_values)

print("最佳 w =", best_w)
print("最小 cost =", best_cost)

plt.figure(figsize=(8, 5))
plt.plot(w_values, cost_values)
plt.axvline(best_w, linestyle="--", label=f"best w = {best_w:.2f}")
plt.xlabel("w")
plt.ylabel("MSE Cost")
plt.title("How Cost Changes with Model Parameter w")
plt.legend()
plt.grid(True)
plt.show()