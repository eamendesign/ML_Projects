import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# dataset
x=np.linspace(-5,5,200)

true_w=2
true_b=-1

prob=1/(1+np.exp(-(true_w*x+true_b)))

y=(prob>0.5).astype(int)

# model
w=0
b=0

lr=0.1
epochs=200

def sigmoid(z):
    return 1/(1+np.exp(-z))

loss_history=[]

for i in range(epochs):

    z=w*x+b
    p=sigmoid(z)

    loss=-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))
    loss_history.append(loss)

    dw=np.mean((p-y)*x)
    db=np.mean(p-y)

    w-=lr*dw
    b-=lr*db
    print(f"i = {i}, w = {w}, b = {b}")

print("trained w,b:",w,b)

plt.plot(loss_history)
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()