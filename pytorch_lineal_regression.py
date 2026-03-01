import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
#  Configuración
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# =========================
#  Crear Dataset personalizado
# =========================

class SimpleDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(-10, 10, 200).reshape(-1, 1)
        self.y = 2 * self.x + 1 + torch.randn(self.x.size()) * 2  # y = 2x + 1 + ruido

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# =========================
#  Definir Modelo
# =========================

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel().to(device)

# =========================
# Función de pérdida y optimizador
# =========================

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# =========================
# Entrenamiento
# =========================

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# =========================
# Evaluación
# =========================

model.eval()
with torch.no_grad():
    test_input = torch.tensor([[4.0]]).to(device)
    prediction = model(test_input)
    print("\nPredicción para x=4:")
    print("Resultado:", prediction.item())

# =========================
#  Mostrar parámetros aprendidos
# =========================

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"\n{name}: {param.data}")
