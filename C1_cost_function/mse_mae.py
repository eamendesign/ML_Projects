import numpy as np

y_true = np.array([3, 5, 2, 7])
y_pred_normal = np.array([2.5, 4.8, 2.2, 8.0])
y_pred_outlier = np.array([2.5, 4.8, 2.2, 20.0])  # 最后一个预测很离谱

def compute_metrics(y_true, y_pred):
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    return mae, mse

mae1, mse1 = compute_metrics(y_true, y_pred_normal)
mae2, mse2 = compute_metrics(y_true, y_pred_outlier)

print("正常预测:")
print("MAE =", mae1, "MSE =", mse1)

print("\n有离群点预测:")
print("MAE =", mae2, "MSE =", mse2)