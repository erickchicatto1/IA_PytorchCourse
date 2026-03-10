import numpy as np

class NeuralNetwork:

    def __init__(self, layers):

        self.layers = layers
        self.weights = []
        self.biases = []

        # inicializar pesos
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    # activación
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # forward
    def forward(self, X):

        activations = [X]
        zs = []

        a = X

        for i in range(len(self.weights)-1):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            a = self.relu(z)
            activations.append(a)

        z = a @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        activations.append(z)

        return activations, zs

    # loss
    def mse(self, y_pred, y):
        return np.mean((y_pred - y)**2)

    # backward
    def backward(self, activations, zs, y):

        grads_w = []
        grads_b = []

        m = y.shape[0]

        # output layer
        delta = (activations[-1] - y) * 2 / m

        dw = activations[-2].T @ delta
        db = np.sum(delta, axis=0, keepdims=True)

        grads_w.append(dw)
        grads_b.append(db)

        # hidden layers
        for i in range(len(self.weights)-2, -1, -1):

            delta = (delta @ self.weights[i+1].T) * self.relu_derivative(zs[i])

            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        return grads_w, grads_b

    # entrenamiento
    def train(self, X, y, epochs=1000, lr=0.001):

        for epoch in range(epochs):

            activations, zs = self.forward(X)

            loss = self.mse(activations[-1], y)

            grads_w, grads_b = self.backward(activations, zs, y)

            for i in range(len(self.weights)):
                self.weights[i] -= lr * grads_w[i]
                self.biases[i] -= lr * grads_b[i]

            if epoch % 100 == 0:
                print("Epoch:", epoch, "Loss:", loss)

    # predicción
    def predict(self, X):

        a = X

        for i in range(len(self.weights)-1):
            a = self.relu(a @ self.weights[i] + self.biases[i])

        return a @ self.weights[-1] + self.biases[-1]


# -----------------------
# Datos de ejemplo
# -----------------------

X = np.array([
    [60,10,1012],
    [65,12,1010],
    [70,8,1008],
    [75,7,1005],
    [80,5,1003]
])

y = np.array([
    [25],
    [26],
    [27],
    [28],
    [30]
])

# crear red
nn = NeuralNetwork([3,8,8,1])

# entrenar
nn.train(X, y, epochs=2000, lr=0.00001)

# predicción
sample = np.array([[72,9,1007]])

print("Predicción:", nn.predict(sample))
