import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10,10,200)

sigmoid = 1/(1+np.exp(-z))

plt.figure(figsize=(7,5))
plt.plot(z,sigmoid,linewidth=2)
plt.axhline(0.5,linestyle="--")
plt.axvline(0,linestyle="--")

plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")
plt.grid(True)

plt.show()