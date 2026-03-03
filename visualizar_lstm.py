import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate  = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate   = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, state):
        h_prev, c_prev = state
        combined = torch.cat((input, h_prev), dim=1)

        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        g = torch.tanh(self.cell_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

# 🔹 Configuración
torch.manual_seed(0)

input_size = 1
hidden_size = 1   # Lo dejamos en 1 para verlo fácil

lstm = LSTMCell(input_size, hidden_size)

# Secuencia simple
sequence = torch.linspace(0, 5, steps=20).unsqueeze(1)

# Estado inicial
h = torch.zeros(1, hidden_size)
c = torch.zeros(1, hidden_size)

h_values = []
c_values = []

for x in sequence:
    x = x.unsqueeze(0)
    h, c = lstm(x, (h, c))
    
    h_values.append(h.item())
    c_values.append(c.item())

# Graficar
plt.figure(figsize=(10,5))

plt.plot(h_values, label="h (estado oculto)")
plt.plot(c_values, label="c (memoria)")
plt.title("Evolución de la LSTM en el tiempo")
plt.xlabel("Paso de tiempo")
plt.ylabel("Valor")
plt.legend()
plt.grid()
plt.show()
