import numpy as np

# 真实标签（1表示正类，0表示负类）
y_true = np.array([1, 0, 1, 1, 0])

# 模型预测“属于正类”的概率
y_pred_prob = np.array([0.9, 0.2, 0.8, 0.4, 0.1])

# 为了避免 log(0)
epsilon = 1e-12
y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)

# Binary Cross-Entropy
bce = -np.mean(
    y_true * np.log(y_pred_prob) +
    (1 - y_true) * np.log(1 - y_pred_prob)
)

print("真实标签:", y_true)
print("预测概率:", y_pred_prob)
print("Binary Cross-Entropy =", bce)