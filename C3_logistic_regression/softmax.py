import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Generate 3-class dataset
# =========================================================
np.random.seed(0)

N = 300
D = 2
K = 3

# create clusters 三个不同类别的二维高斯分布数据

X1 = np.random.randn(N//3,2) + np.array([-2,0]) # 
X2 = np.random.randn(N//3,2) + np.array([2,2])
X3 = np.random.randn(N//3,2) + np.array([3,-2])

X = np.vstack([X1,X2,X3])

y = np.array([0]*(N//3) + [1]*(N//3) + [2]*(N//3))


# =========================================================
# 2. One-hot encoding
# =========================================================
Y = np.zeros((N,K))

for i in range(N):
    Y[i,y[i]] = 1


# =========================================================
# 3. Softmax function
# =========================================================
def softmax(Z):

    Z = Z - np.max(Z,axis=1,keepdims=True) # 减去样本每一行的最大值, 防止指数爆炸的关键数值稳定技巧

    expZ = np.exp(Z)

    return expZ / np.sum(expZ,axis=1,keepdims=True)


# =========================================================
# 4. Initialize parameters
# =========================================================
W = np.random.randn(D,K) * 0.01
b = np.zeros((1,K))

lr = 0.1
epochs = 500

loss_history = []


# =========================================================
# 5. Training loop
# =========================================================
for epoch in range(epochs):

    # forward
    scores = X @ W + b

    probs = softmax(scores)

    # cross entropy loss
    loss = -np.mean(np.sum(Y*np.log(probs+1e-9),axis=1))

    loss_history.append(loss)

    # gradient
    dZ = (probs - Y) / N

    dW = X.T @ dZ

    db = np.sum(dZ,axis=0,keepdims=True)

    # update
    W -= lr * dW
    b -= lr * db


print("Final loss:",loss_history[-1])


# =========================================================
# 6. Plot loss curve
# =========================================================
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy")
plt.show()


# =========================================================
# 7. Decision boundary visualization
# =========================================================

x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1

xx,yy = np.meshgrid(
    np.linspace(x_min,x_max,300),
    np.linspace(y_min,y_max,300)
)

grid = np.c_[xx.ravel(),yy.ravel()]

scores = grid @ W + b
probs = softmax(scores)

pred = np.argmax(probs,axis=1)
pred = pred.reshape(xx.shape)


plt.figure(figsize=(8,6))

plt.contourf(xx,yy,pred,alpha=0.3,cmap="Set1")

plt.scatter(X[:,0],X[:,1],c=y,cmap="Set1",edgecolor="k")

plt.title("Softmax Regression Decision Boundary")

plt.xlabel("x1")
plt.ylabel("x2")

plt.show()