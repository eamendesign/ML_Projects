import numpy as np

# 原始数据：10个样本
X = np.array([
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 5],
    [5, 4],
    [6, 6],
    [7, 8],
    [8, 7],
    [9, 9],
    [10, 10]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

def bootstrap_sample(X, y):
    n = len(X)
    indices = np.random.choice(n, size=n, replace=True)  # 有放回抽样
    return X[indices], y[indices], indices

# 模拟3棵树的训练数据
for i in range(3):
    Xb, yb, idx = bootstrap_sample(X, y)
    print(f"\n树 {i+1} 的bootstrap样本索引:")
    print(idx)
    print("对应标签:", yb)