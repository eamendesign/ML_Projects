import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 构造两类简单数据
# =========================
X_pos = np.array([
    [2, 3],
    [3, 3],
    [2.5, 4],
    [3.5, 4]
])

X_neg = np.array([
    [6, 2],
    [7, 3],
    [6.5, 4],
    [8, 3]
])

# 假设我们已经有一个分类超平面：w^T x + b = 0
# 这里只是为了演示最大间隔的含义，不是训练出来的
w = np.array([1.0, -0.2])
b = -4.5

# 间隔线:
# w^T x + b = +1
# w^T x + b = -1

def line_y(x, w, b, c=0):
    # w1*x + w2*y + b = c
    # => y = (c - w1*x - b)/w2
    return (c - w[0]*x - b) / w[1]

x_vals = np.linspace(0, 10, 200)
y_decision = line_y(x_vals, w, b, c=0)
y_margin1  = line_y(x_vals, w, b, c=1)
y_margin2  = line_y(x_vals, w, b, c=-1)

# 假设这些点最靠近间隔，是支持向量
support_vectors = np.array([
    [3.5, 4],
    [6, 2]
])

# =========================
# 2. 画图
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(X_pos[:, 0], X_pos[:, 1], label="Class +1")
plt.scatter(X_neg[:, 0], X_neg[:, 1], label="Class -1")

plt.plot(x_vals, y_decision, label="Decision boundary: w·x+b=0")
plt.plot(x_vals, y_margin1, '--', label="Margin: w·x+b=1")
plt.plot(x_vals, y_margin2, '--', label="Margin: w·x+b=-1")

plt.scatter(
    support_vectors[:, 0], support_vectors[:, 1],
    s=200, facecolors='none', edgecolors='red', linewidths=2,
    label="Support vectors"
)

plt.xlim(0, 10)
plt.ylim(0, 6)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Example 1: Maximum Margin Intuition")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 3. 打印间隔宽度
# =========================
margin = 2 / np.linalg.norm(w)
print("w =", w)
print("b =", b)
print("Margin width = 2 / ||w|| =", margin)