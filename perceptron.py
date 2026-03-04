import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, epochs=100):
        # Inicializamos pesos en cero (o valores aleatorios pequeños)
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        # Suma ponderada: (entradas * pesos) + bias
        summation = np.dot(inputs, self.weights) + self.bias
        # Función de activación: Escalón (Heaviside)
        return 1 if summation > 0 else 0

    def train(self, training_data, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                # Regla de actualización del perceptrón
                error = label - prediction
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error

# --- Ejemplo de uso: Compuerta AND ---
# Entradas (X1, X2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Etiquetas (Resultados esperados de AND)
y = np.array([0, 0, 0, 1])

modelo = Perceptron(n_inputs=2)
modelo.train(X, y)

# Prueba del modelo
print("Predicciones para AND:")
for p in X:
    print(f"Entrada {p} -> Predicción: {modelo.predict(p)}")
