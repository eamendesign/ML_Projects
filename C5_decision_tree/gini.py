import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 构造一个简单数据集
# -----------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# -----------------------------
# 2. 定义 Gini impurity
# -----------------------------
def gini(labels):
    if len(labels) == 0:
        return 0
    p0 = np.sum(labels == 0) / len(labels)
    p1 = np.sum(labels == 1) / len(labels)
    return 1 - p0**2 - p1**2

# -----------------------------
# 3. 计算某个切分点的加权 Gini
# -----------------------------
def weighted_gini_split(X, y, threshold):
    left_mask = X <= threshold
    right_mask = X > threshold
    
    y_left = y[left_mask]
    y_right = y[right_mask]
    
    g_left = gini(y_left)
    g_right = gini(y_right)
    
    weighted_g = (len(y_left) / len(y)) * g_left + (len(y_right) / len(y)) * g_right
    return weighted_g, y_left, y_right

# -----------------------------
# 4. 枚举所有候选切分点
# -----------------------------
thresholds = [(X[i] + X[i+1]) / 2 for i in range(len(X)-1)]

best_threshold = None
best_gini = float('inf')

print("候选切分点的加权 Gini：")
for t in thresholds:
    wg, y_left, y_right = weighted_gini_split(X, y, t)
    print(f"threshold = {t:.1f}, weighted gini = {wg:.3f}")
    if wg < best_gini:
        best_gini = wg
        best_threshold = t

print("\n最优切分点：", best_threshold)
print("最小加权 Gini：", best_gini)

# -----------------------------
# 5. 可视化
# -----------------------------
plt.figure(figsize=(8, 4))
for xi, yi in zip(X, y):
    if yi == 0:
        plt.scatter(xi, yi, s=120, label='Class 0' if xi == X[0] else "", marker='o')
    else:
        plt.scatter(xi, yi, s=120, label='Class 1' if xi == X[4] else "", marker='s')

plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best split = {best_threshold}')
plt.yticks([0, 1], ['Class 0', 'Class 1'])
plt.xlabel("Feature X")
plt.title("Decision Tree: Best Split on 1D Data")
plt.legend()
plt.grid(True)
plt.show()