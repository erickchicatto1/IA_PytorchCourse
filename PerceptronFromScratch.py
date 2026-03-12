import numpy as np
import matplotlib.pyplot as plt

class PerceptronSimple:
    def __init__(self, input_size):
        # Un solo peso y un solo bias
        self.w = np.random.randn(input_size, 1) * 0.1
        self.b = np.zeros((1, 1))

    def predict(self, X):
        # y = X * w + b (Regresión lineal pura)
        return X @ self.w + self.b

    def train(self, X, y, epochs=200, lr=0.1):
        m = X.shape[0]
        # Lista para guardar la pérdida (Loss) y graficarla después
        losses = [] 

        for epoch in range(epochs):
            # Forward: predecir
            y_pred = self.predict(X)
            
            # Cálculo del Error (MSE)
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Backpropagation (Gradientes simplificados)
            error = y_pred - y
            dw = (X.T @ error) * 2 / m
            db = np.sum(error) * 2 / m
            
            # Actualizar parámetros (Descenso de gradiente)
            self.w -= lr * dw
            self.b -= lr * db
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}")
        
        return losses # Devolvemos la lista de pérdidas

# ----------------------------------------------------
# 1. Generar datos sintéticos para entrenar
# ----------------------------------------------------
# Queremos que aprenda la relación y = 3.5 * x + 1.2
m = 100 # Número de muestras
X_train = np.linspace(-1, 1, m).reshape(-1, 1)
# Añadimos un poco de "ruido" aleatorio para que sea más realista
ruido = np.random.randn(m, 1) * 0.2
y_train = 3.5 * X_train + 1.2 + ruido

# ----------------------------------------------------
# 2. Crear y entrenar el Perceptrón
# ----------------------------------------------------
p = PerceptronSimple(input_size=1)
# Obtenemos la lista de pérdidas (losses)
losses = p.train(X_train, y_train, epochs=200, lr=0.1)

# ----------------------------------------------------
# 3. Predicción Final
# ----------------------------------------------------
y_pred_final = p.predict(X_train)

# ----------------------------------------------------
# 4. Graficar Resultados con Matplotlib
# ----------------------------------------------------
# Crear una figura con 2 subgráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Los datos y la predicción (El "Ajuste de la Línea")
ax1.scatter(X_train, y_train, color='blue', alpha=0.5, label='Datos Reales (con Ruido)')
ax1.plot(X_train, y_pred_final, color='red', linewidth=3, label='Predicción del Perceptrón')
ax1.set_title("Ajuste del Perceptrón a los Datos")
ax1.set_xlabel("Entrada (X)")
ax1.set_ylabel("Salida (Y)")
ax1.legend()
ax1.grid(True, linestyle='--')

# Gráfico 2: La Evolución de la Pérdida (Loss)
ax2.plot(losses, color='purple', linewidth=2)
ax2.set_title("Evolución de la Pérdida (Loss) durante el Entrenamiento")
ax2.set_xlabel("Época")
ax2.set_ylabel("Pérdida (MSE)")
ax2.grid(True, linestyle='--')

# Mostrar todo
plt.tight_layout()
plt.show()

# Imprimir los valores aprendidos
print(f"\nValores aprendidos:")
print(f"Peso (w): {p.w[0][0]:.3f} (Objetivo: 3.5)")
print(f"Bias (b): {p.b[0][0]:.3f} (Objetivo: 1.2)")
