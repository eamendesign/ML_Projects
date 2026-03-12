import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 生成二维数据
# =========================
np.random.seed(42)

n = 40
X_pos = np.random.randn(n, 2) * 0.8 + np.array([2, 2])
X_neg = np.random.randn(n, 2) * 0.8 + np.array([5, 5])

X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n), -np.ones(n)])   # +1 / -1

# =========================
# 2. 初始化参数
# =========================
w = np.zeros(2)
b = 0.0

lr = 0.001
C = 1.0
epochs = 3000

loss_history = []

# =========================
# 3. 训练：次梯度下降
# =========================
for epoch in range(epochs):
    dw = np.zeros_like(w)
    db = 0.0
    loss = 0.5 * np.dot(w, w)

    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        condition = yi * (np.dot(w, xi) + b)

        if condition >= 1:
            # 只来自正则项
            dw += w / len(X)
        else:
            # 正则项 + hinge loss
            dw += (w - C * yi * xi) / len(X)
            db += (-C * yi) / len(X)
            loss += C * (1 - condition)

    w -= lr * dw
    b -= lr * db
    loss_history.append(loss)

print("训练完成")
print("w =", w)
print("b =", b)

# =========================
# 4. 预测函数
# =========================
def predict(X, w, b):
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)

y_pred = predict(X, w, b)
acc = np.mean(y_pred == y)
print("Accuracy =", acc)

# =========================
# 5. 找支持向量（近似）
# =========================
scores = y * (X @ w + b)
support_mask = np.abs(scores - 1) < 0.08
support_vectors = X[support_mask]

# =========================
# 6. 画分类边界
# =========================
def line_y(x, w, b, c=0):
    return (c - w[0] * x - b) / w[1]

x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
y_decision = line_y(x_vals, w, b, c=0)
y_margin1  = line_y(x_vals, w, b, c=1)
y_margin2  = line_y(x_vals, w, b, c=-1)

plt.figure(figsize=(8, 6))
plt.scatter(X[y==1, 0], X[y==1, 1], label="Class +1")
plt.scatter(X[y==-1, 0], X[y==-1, 1], label="Class -1")

plt.plot(x_vals, y_decision, label="Decision boundary")
plt.plot(x_vals, y_margin1, '--', label="Margin +1")
plt.plot(x_vals, y_margin2, '--', label="Margin -1")

if len(support_vectors) > 0:
    plt.scatter(
        support_vectors[:, 0], support_vectors[:, 1],
        s=180, facecolors='none', edgecolors='red', linewidths=2,
        label="Approx. support vectors"
    )

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Example 2: Linear Soft-Margin SVM (No sklearn)")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 7. 画损失函数下降过程
# =========================
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()