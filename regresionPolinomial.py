import numpy as np
import math
import matplotlib.pyplot as plt

# Datos
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Pesos iniciales aleatorios
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
loss_history = []

plt.ion()  # modo interactivo para actualizar gráficos

for t in range(2000):

    # Forward pass
    y_pred = a + b*x + c*x**2 + d*x**3

    # Loss
    loss = np.square(y_pred - y).sum()
    loss_history.append(loss)

    if t % 200 == 0:

        plt.clf()

        plt.subplot(1,2,1)
        plt.title(f"Aproximación en iteración {t}")
        plt.plot(x, y, label="sin(x)")
        plt.plot(x, y_pred, label="polinomio")
        plt.legend()

        plt.subplot(1,2,2)
        plt.title("Loss")
        plt.plot(loss_history)

        plt.pause(0.1)

    # Backprop
    grad_y_pred = 2.0 * (y_pred - y)

    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # Update
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


plt.ioff()
plt.show()

print(f"Resultado final:")
print(f"y = {a} + {b}x + {c}x² + {d}x³")
