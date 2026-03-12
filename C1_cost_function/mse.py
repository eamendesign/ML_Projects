import numpy as np

# 真实值
y_true = np.array([3, 5, 2, 7])

# 模型预测值
y_pred = np.array([2.5, 4.8, 2.2, 8.0])

# 误差
errors = y_pred - y_true

# 平方误差
squared_errors = errors ** 2

# MSE
mse = np.mean(squared_errors)

print("真实值:", y_true)
print("预测值:", y_pred)
print("误差:", errors)
print("平方误差:", squared_errors)
print("MSE =", mse)