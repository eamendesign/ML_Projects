import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# 为了画光滑曲线
X_plot = np.linspace(0, 2 * np.pi, 400).reshape(-1, 1)
y_true = np.sin(X_plot)

# ----------------------------
# 2. 定义两个模型
# ----------------------------
model_simple = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

model_complex = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)

# 训练模型
model_simple.fit(X_train, y_train)
model_complex.fit(X_train, y_train)

# 预测
y_train_pred_simple = model_simple.predict(X_train)
y_test_pred_simple = model_simple.predict(X_test)

y_train_pred_complex = model_complex.predict(X_train)
y_test_pred_complex = model_complex.predict(X_test)

# 计算误差
train_mse_simple = mean_squared_error(y_train, y_train_pred_simple)
test_mse_simple = mean_squared_error(y_test, y_test_pred_simple)

train_mse_complex = mean_squared_error(y_train, y_train_pred_complex)
test_mse_complex = mean_squared_error(y_test, y_test_pred_complex)

print("=== 3阶多项式模型 ===")
print(f"训练集 MSE: {train_mse_simple:.4f}")
print(f"测试集 MSE: {test_mse_simple:.4f}")

print("\n=== 15阶多项式模型 ===")
print(f"训练集 MSE: {train_mse_complex:.4f}")
print(f"测试集 MSE: {test_mse_complex:.4f}")

# ----------------------------
# 3. 可视化
# ----------------------------
plt.figure(figsize=(12, 5))

# 左图：简单模型
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Train data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_plot, y_true, 'k--', label='True function: sin(x)')
plt.plot(X_plot, model_simple.predict(X_plot), color='red', label='Degree=3')
plt.title("Good Fit / Slight Underfit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 右图：复杂模型
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, color='blue', label='Train data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_plot, y_true, 'k--', label='True function: sin(x)')
plt.plot(X_plot, model_complex.predict(X_plot), color='red', label='Degree=15')
plt.title("Overfitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()