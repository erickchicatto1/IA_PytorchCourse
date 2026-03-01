import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================
#  Configuración de dispositivo
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# =====================================================
# Crear Dataset personalizado usando tensores
# =====================================================
class CircleDataset(Dataset):
    def __init__(self, n_samples=1000):
        # Crear puntos aleatorios 2D
        self.X = torch.rand(n_samples, 2) * 2 - 1  # rango [-1,1]
        
        # Etiquetas: dentro o fuera del círculo
        self.y = (self.X[:, 0]**2 + self.X[:, 1]**2 > 1).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CircleDataset(n_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =====================================================
#  Definir Red Neuronal
# =====================================================
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = NeuralNet().to(device)

# =====================================================
# Loss y Optimizador
# =====================================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =====================================================
# Entrenamiento
# =====================================================
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

# =====================================================
# Evaluación
# =====================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        outputs = model(X_batch)
        predicted = (outputs > 0.5).float()
        
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print("\nAccuracy:", accuracy)

# =====================================================
#  Información del modelo (what, what, where)
# =====================================================
print("\nInformación del modelo:")
for name, param in model.named_parameters():
    print(f"{name} -> Shape: {param.shape}, Device: {param.device}, Dtype: {param.dtype}")
