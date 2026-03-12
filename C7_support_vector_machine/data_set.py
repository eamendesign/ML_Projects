import numpy as np

def generate_svm_data(n=40,
                      center_pos=(2,2),
                      center_neg=(5,5),
                      noise=1.0,
                      seed=None):
    """
    Generate a simple 2D dataset for SVM
    
    Parameters
    ----------
    n : number of samples per class
    center_pos : center of positive class
    center_neg : center of negative class
    noise : standard deviation of data
    seed : random seed
    
    Returns
    -------
    X : feature matrix
    y : labels (+1,-1)
    """

    if seed is not None:
        np.random.seed(seed)

    # 正类
    X_pos = noise * np.random.randn(n,2) + np.array(center_pos)

    # 负类
    X_neg = noise * np.random.randn(n,2) + np.array(center_neg)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n), -np.ones(n)])

    return X, y

import matplotlib.pyplot as plt

X, y = generate_svm_data(center_pos=(1,1),
                         center_neg=(4,4),
                         seed=0)

plt.scatter(X[y==1][:,0], X[y==1][:,1], label="+1")
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], label="-1")

# 添加中心点标记（可选）
plt.scatter(1, 1, c='black', marker='x', s=100, label='Center 0')
plt.scatter(4, 4, c='black', marker='x', s=100, label='Center 1')

plt.legend()
plt.grid(True)
plt.show()
