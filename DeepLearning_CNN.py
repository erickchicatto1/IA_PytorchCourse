# =========================
# 1. IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================
# 2. DATASET (MNIST)
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =========================
# 3. VISUALIZAR DATOS
# =========================
images, labels = next(iter(dataloader))

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(labels[i].item())
    axes[i].axis('off')
plt.show()

# =========================
# 4. MODELO CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 3)     # (1,28,28) -> (8,26,26)
        self.conv2 = nn.Conv2d(8, 16, 3)    # -> (16,11,11)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # -> (8,13,13)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # -> (16,5,5)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = SimpleCNN()

# =========================
# 5. LOSS Y OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 6. ENTRENAMIENTO
# =========================
losses = []

epochs = 3

for epoch in range(epochs):
    for images, labels in dataloader:
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# =========================
# 7. GRAFICAR LOSS
# =========================
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Iteraciones")
plt.ylabel("Loss")
plt.show()

# =========================
# 8. PROBAR EL MODELO
# =========================
images, labels = next(iter(dataloader))
outputs = model(images)

# predicción
_, preds = torch.max(outputs, 1)

# mostrar resultados
fig, axes = plt.subplots(1, 5, figsize=(10,3))
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f"Real: {labels[i].item()}\nPred: {preds[i].item()}")
    axes[i].axis('off')

plt.show()
