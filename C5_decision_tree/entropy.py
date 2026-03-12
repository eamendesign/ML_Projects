import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

def entropy(labels):
    if len(labels) == 0:
        return 0
    classes, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-12))

def information_gain(X, y, threshold):
    parent_entropy = entropy(y)
    
    left_mask = X <= threshold
    right_mask = X > threshold
    
    y_left = y[left_mask]
    y_right = y[right_mask]
    
    child_entropy = (len(y_left)/len(y))*entropy(y_left) + (len(y_right)/len(y))*entropy(y_right)
    
    ig = parent_entropy - child_entropy
    return ig

thresholds = [(X[i] + X[i+1]) / 2 for i in range(len(X)-1)]

best_threshold = None
best_entropy = float('inf')

print("各个切分点的信息增益：")
for t in thresholds:
    ig = information_gain(X, y, t)
    print(f"threshold = {t:.1f}, information gain = {ig:.4f}")

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