import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 1. Generate synthetic data
# ==========================

np.random.seed(0)

n=200

# features
size=np.random.normal(100,20,n)
bedroom=np.random.randint(1,5,n)

# true model
true_w1=3000
true_w2=20000
true_b=50000

noise=np.random.normal(0,10000,n)

price=true_w1*size+true_w2*bedroom+true_b+noise

# stack features
X=np.column_stack([size,bedroom])
y=price

# ==========================
# 2. Initialize parameters
# ==========================

w=np.zeros(2)
b=0

lr=1e-6
epochs=200

loss_history=[]

# ==========================
# 3. Gradient Descent
# ==========================

for i in range(epochs):

    y_pred=X@w+b

    error=y_pred-y

    loss=np.mean(error**2)

    loss_history.append(loss)

    dw=(2/n)*(X.T@error)

    db=(2/n)*np.sum(error)

    w=w-lr*dw
    b=b-lr*db

print(f"learned parameters: w = {w} , b = {b}")

# ==========================
# 4. Plot loss
# ==========================

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

ax.scatter(size,bedroom,price)

size_grid,bed_grid=np.meshgrid(
np.linspace(size.min(),size.max(),20),
np.linspace(bedroom.min(),bedroom.max(),20))

price_pred=w[0]*size_grid+w[1]*bed_grid+b

ax.plot_surface(size_grid,bed_grid,price_pred,alpha=0.5)
w_str = np.array2string(w, precision=2, separator=', ')
ax.set_title(f"learned parameters: w = {w_str} , b = {b:.2f}")
ax.set_xlabel("Size")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")

plt.show()