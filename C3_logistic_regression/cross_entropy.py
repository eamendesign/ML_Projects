import numpy as np
import matplotlib.pyplot as plt

p=np.linspace(0.01,0.99,200)

loss1=-np.log(p)
loss0=-np.log(1-p)

plt.plot(p,loss1,label="y=1")
plt.plot(p,loss0,label="y=0")

plt.xlabel("Predicted probability")
plt.ylabel("Loss")

plt.title("Cross Entropy Loss")

plt.legend()

plt.grid()

plt.show()