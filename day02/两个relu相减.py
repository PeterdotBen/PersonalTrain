import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-1, 5, 400)
y1 = relu(x - 1)
y2 = relu(x - 3)
y_diff = y1 - y2

plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='ReLU(x - 1)', linestyle='--')
plt.plot(x, y2, label='ReLU(x - 3)', linestyle='--')
plt.plot(x, y_diff, label='ReLU(x - 1) - ReLU(x - 3)', linewidth=2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.title('Two ReLU Functions and Their Difference')
plt.xlabel('x')
plt.ylabel('y')
plt.show()