import torch
import torch.nn as nn
import torch.optim as optim

# Definición de la red
class RedNeuronal(nn.Module):
    def __init__(self, entrada, oculta, salida):
        super(RedNeuronal, self).__init__()
        self.capa1 = nn.Linear(entrada, oculta)
        self.relu = nn.ReLU()
        self.capa2 = nn.Linear(oculta, salida)

    def forward(self, x):
        x = self.capa1(x)
        x = self.relu(x)
        x = self.capa2(x)
        return x

# Hiperparámetros
tamano_entrada = 10      # número de características
tamano_oculto = 32
tamano_salida = 2        # número de clases

# Instanciar modelo, función de pérdida y optimizador
modelo = RedNeuronal(tamano_entrada, tamano_oculto, tamano_salida)
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Datos de ejemplo (aleatorios, reemplazar con tus datos reales)
X = torch.randn(100, tamano_entrada)
y = torch.randint(0, tamano_salida, (100,))

# Entrenamiento
epocas = 50
for epoca in range(epocas):
    optimizador.zero_grad()
    salidas = modelo(X)
    perdida = criterio(salidas, y)
    perdida.backward()
    optimizador.step()

    if (epoca + 1) % 10 == 0:
        print(f"Época [{epoca+1}/{epocas}], Pérdida: {perdida.item():.4f}")
