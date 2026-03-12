import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
np.random.seed(0)
x1 = np.random.normal(2, 1, 50) # 第一类点，均值2
x2 = np.random.normal(6, 1, 50) # 第二类点，均值6
x = np.concatenate([x1, x2])
y = np.concatenate([np.zeros(50), np.ones(50)])

# 2. 逻辑回归核心：Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 设定权重和偏置
w = 2
b = -8

# 计算决策边界 (Decision Boundary): 当 w*x + b = 0 时，prob = 0.5
# 即 x = -b / w
decision_boundary = -b / w

# 3. 绘图准备
x_line = np.linspace(0, 10, 200)
prob = sigmoid(w * x_line + b)

plt.figure(figsize=(10, 6))

# 画出原始样本点
plt.scatter(x1, [0]*50, color='blue', alpha=0.5, label="Class 0 (Actual)")
plt.scatter(x2, [1]*50, color='green', alpha=0.5, label="Class 1 (Actual)")

# 画出逻辑回归生成的 S 曲线（预测概率）
plt.plot(x_line, prob, color='red', linewidth=2, label="Logistic Curve (Probability)")

# 画出 0.5 概率阈值线
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

# 画出决策边界垂直线
plt.axvline(decision_boundary, color='black', linestyle=':', label=f'Decision Boundary (x={decision_boundary})')

# 填充背景色来展示分类区域
plt.fill_between(x_line, 0, 1, where=(x_line >= decision_boundary), color='green', alpha=0.1)
plt.fill_between(x_line, 0, 1, where=(x_line < decision_boundary), color='blue', alpha=0.1)

# 文字说明
plt.text(decision_boundary+0.2, 0.6, 'Predict Class 1', fontsize=12, color='darkgreen')
plt.text(decision_boundary-2.5, 0.4, 'Predict Class 0', fontsize=12, color='darkblue')

# 标签和标题
plt.xlabel("Feature value (x)")
plt.ylabel("Probability / Class")
plt.title("How Logistic Regression Makes Decisions", fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.show()