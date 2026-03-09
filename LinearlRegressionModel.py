import torch
from torch import nn

# Crear datos de ejemplo
# y = 2x + 1
X = torch.arange(0, 10, dtype=torch.float).unsqueeze(dim=1)
y = 2 * X + 1

# Dividir datos en entrenamiento y prueba
train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]


# Crear el modelo de regresión lineal
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # peso (pendiente)
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

        # bias (intersección)
        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

    # función forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# Crear el modelo
model = LinearRegressionModel()

# Mostrar parámetros iniciales
print("Parámetros iniciales:")
print(model.state_dict())


# Definir función de pérdida
loss_fn = nn.L1Loss()

# Definir optimizador
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=0.01
)

# Número de épocas
epochs = 200

# Entrenamiento
for epoch in range(epochs):

    # Modo entrenamiento
    model.train()

    # Predicción
    y_pred = model(X_train)

    # Calcular pérdida
    loss = loss_fn(y_pred, y_train)

    # Resetear gradientes
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Actualizar parámetros
    optimizer.step()

    # Evaluación
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    # Imprimir progreso
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")


# Mostrar parámetros finales
print("\nParámetros finales:")
print(model.state_dict())


# Hacer predicciones
with torch.inference_mode():
    predictions = model(X_test)

print("\nPredicciones:")
print(predictions)

print("\nValores reales:")
print(y_test)
