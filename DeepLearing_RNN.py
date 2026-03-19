#La RNN aprendera a predecir una onda señal seno 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. MODELO
# =========================
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :]  # último paso
        out = self.fc(out)
        return out

model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)

# =========================
# 2. DATOS (onda seno)
# =========================
t = np.linspace(0, 100, 1000)
data = np.sin(t)

# visualizar datos originales
plt.plot(t[:100], data[:100])
plt.title("Señal original (seno)")
plt.show()

# =========================
# 3. CREAR SECUENCIAS
# =========================
def create_sequences(data, seq_length):
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)

seq_length = 20
X, y = create_sequences(data, seq_length)

# convertir a tensor
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# =========================
# 4. ENTRENAMIENTO
# =========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []

for epoch in range(10):
    output = model(X)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# =========================
# 5. GRAFICAR LOSS
# =========================
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# =========================
# 6. PREDICCIONES
# =========================
with torch.no_grad():
    preds = model(X).squeeze().numpy()

# =========================
# 7. VISUALIZAR RESULTADOS
# =========================
plt.figure(figsize=(10,5))

plt.plot(y.numpy()[:200], label="Real")
plt.plot(preds[:200], label="Predicción")

plt.legend()
plt.title("RNN aprendiendo la onda seno")
plt.show()
