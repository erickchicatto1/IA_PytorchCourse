"""
GAN Básica en PyTorch
=====================
Objetivo: G aprende a generar muestras de una distribución gaussiana N(4, 1.5)
partiendo solo de ruido uniforme U(-1, 1).

Conceptos que cubre este ejemplo:
  - Arquitectura de G y D
  - Función de pérdida BCE (Binary Cross-Entropy)
  - El juego minimax: D maximiza, G minimiza
  - Por qué G usa log D(G(z)) en lugar de log(1 - D(G(z)))
  - Visualización de la convergencia
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 0. Reproducibilidad
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. Distribución REAL que queremos aprender
# ─────────────────────────────────────────────
# La "verdad" que D conoce y G intenta imitar.
REAL_MEAN = 4.0
REAL_STD  = 1.5

def sample_real(batch_size: int) -> torch.Tensor:
    """Muestras de la distribución real N(4, 1.5)."""
    return torch.FloatTensor(batch_size, 1).normal_(REAL_MEAN, REAL_STD)

# ─────────────────────────────────────────────
# 2. Distribución de RUIDO (entrada de G)
# ─────────────────────────────────────────────
NOISE_DIM = 1  # dimensión del ruido latente z

def sample_noise(batch_size: int) -> torch.Tensor:
    """Ruido uniforme U(-1, 1) — punto de partida del generador."""
    return torch.FloatTensor(batch_size, NOISE_DIM).uniform_(-1, 1)

# ─────────────────────────────────────────────
# 3. GENERADOR G: z ∈ R¹ → x̃ ∈ R¹
# ─────────────────────────────────────────────
# G toma ruido y lo transforma en datos falsos.
# Usamos Tanh en la salida para mantener valores acotados,
# pero no limitamos demasiado el rango para permitir aprender
# la media ~4 de la distribución real.
class Generator(nn.Module):
    def __init__(self, noise_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            # Sin activación final: G puede producir cualquier valor real.
            # Si usáramos Tanh aquí, limitaríamos la salida a (-1, 1)
            # y G nunca podría aprender la media = 4.
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# ─────────────────────────────────────────────
# 4. DISCRIMINADOR D: x ∈ R¹ → p ∈ (0, 1)
# ─────────────────────────────────────────────
# D es un clasificador binario: ¿real o falso?
# Sigmoid en la salida garantiza que D(x) ∈ (0, 1).
class Discriminator(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LeakyReLU(0.2),   # LeakyReLU evita neuronas muertas en D
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),         # salida ∈ (0, 1) = probabilidad de ser real
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ─────────────────────────────────────────────
# 5. Inicializar redes y optimizadores
# ─────────────────────────────────────────────
G = Generator()
D = Discriminator()

# Cada red tiene su propio optimizador — entrenan de forma independiente.
lr = 1e-3
opt_G = optim.Adam(G.parameters(), lr=lr)
opt_D = optim.Adam(D.parameters(), lr=lr)

# Binary Cross-Entropy: pérdida estándar para clasificación binaria.
# Para D: BCE(D(x), 1) = -log D(x)       → minimizar ≡ maximizar log D(x)
#         BCE(D(G(z)), 0) = -log(1-D(G(z)))
# Para G: BCE(D(G(z)), 1) = -log D(G(z)) → maximizar log D(G(z)) ← truco saturación
criterion = nn.BCELoss()

# ─────────────────────────────────────────────
# 6. LOOP DE ENTRENAMIENTO
# ─────────────────────────────────────────────
EPOCHS     = 5000
BATCH_SIZE = 128

# Guardar métricas para visualización
history = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": []}
snapshots = {}  # distribución de G en distintos epochs

print(f"{'Epoch':>6}  {'Loss D':>8}  {'Loss G':>8}  {'D(real)':>8}  {'D(G(z))':>8}")
print("─" * 54)

for epoch in range(1, EPOCHS + 1):

    # ── Fase 1: Entrenar D ──────────────────────────────────
    # D quiere: D(real) → 1  y  D(G(z)) → 0
    # G está CONGELADO en esta fase (no llamamos opt_G.step()).

    D.train(); G.train()

    real = sample_real(BATCH_SIZE)
    z    = sample_noise(BATCH_SIZE)
    fake = G(z).detach()   # .detach() desconecta G del grafo — no fluyen gradientes a G

    # Etiquetas: 1 = real, 0 = falso
    ones  = torch.ones(BATCH_SIZE, 1)
    zeros = torch.zeros(BATCH_SIZE, 1)

    loss_D_real = criterion(D(real), ones)   # -log D(x)
    loss_D_fake = criterion(D(fake), zeros)  # -log(1 - D(G(z)))
    loss_D = loss_D_real + loss_D_fake       # pérdida total de D

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # ── Fase 2: Entrenar G ──────────────────────────────────
    # G quiere engañar a D: que D(G(z)) → 1
    # D está CONGELADO en esta fase (no llamamos opt_D.step()).

    z    = sample_noise(BATCH_SIZE)
    fake = G(z)  # sin .detach() — ahora SÍ necesitamos gradientes en G

    # Truco no-saturación: G maximiza log D(G(z))
    # ≡ minimiza BCE(D(G(z)), 1) = -log D(G(z))
    # En vez de minimizar log(1 - D(G(z))), que se satura al inicio.
    loss_G = criterion(D(fake), ones)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    # ── Registro ────────────────────────────────────────────
    with torch.no_grad():
        d_real = D(real).mean().item()
        d_fake = D(G(sample_noise(BATCH_SIZE))).mean().item()

    history["loss_D"].append(loss_D.item())
    history["loss_G"].append(loss_G.item())
    history["D_real"].append(d_real)
    history["D_fake"].append(d_fake)

    # Guardar snapshot de la distribución generada
    if epoch in (1, 100, 500, 1000, 2000, 5000):
        with torch.no_grad():
            s = G(sample_noise(1000)).numpy().flatten()
        snapshots[epoch] = s

    if epoch % 500 == 0 or epoch == 1:
        print(f"{epoch:>6}  {loss_D.item():>8.4f}  {loss_G.item():>8.4f}"
              f"  {d_real:>8.4f}  {d_fake:>8.4f}")

print("\n✓ Entrenamiento completado.")

# ─────────────────────────────────────────────
# 7. VISUALIZACIÓN
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("GAN Básica — aprendiendo N(4, 1.5)", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 7a. Evolución de pérdidas ──────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(history["loss_D"], label="Loss D", alpha=0.7, color="#534AB7")
ax1.plot(history["loss_G"], label="Loss G", alpha=0.7, color="#D85A30")
ax1.set_title("Pérdidas durante el entrenamiento")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
ax1.legend(); ax1.grid(alpha=0.3)

# ── 7b. Evolución de D(x) ─────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(history["D_real"], label="D(real)", alpha=0.8, color="#0F6E56")
ax2.plot(history["D_fake"], label="D(G(z))", alpha=0.8, color="#993C1D")
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 (equilibrio)")
ax2.set_title("Salida de D a lo largo del tiempo")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("D(x)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# ── 7c. Snapshots de la distribución generada ─
snap_epochs = [1, 100, 500, 1000, 2000, 5000]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(snap_epochs)))
x_range = np.linspace(-4, 12, 200)
real_pdf = (1 / (REAL_STD * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x_range - REAL_MEAN) / REAL_STD) ** 2
)

ax3 = fig.add_subplot(gs[1, :])
ax3.plot(x_range, real_pdf, "k--", linewidth=2, label="Distribución real N(4, 1.5)", zorder=10)

for i, ep in enumerate(snap_epochs):
    ax3.hist(
        snapshots[ep], bins=40, density=True, alpha=0.45,
        color=colors[i], label=f"G epoch {ep}"
    )

ax3.set_title("Evolución de la distribución generada (histogramas) vs real (línea)")
ax3.set_xlabel("Valor"); ax3.set_ylabel("Densidad")
ax3.legend(fontsize=8, ncol=4); ax3.grid(alpha=0.3)

plt.savefig("gan_basica_resultados.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figura guardada como 'gan_basica_resultados.png'")

# ─────────────────────────────────────────────
# 8. COMPROBACIÓN FINAL
# ─────────────────────────────────────────────
with torch.no_grad():
    final_samples = G(sample_noise(10000)).numpy().flatten()

print(f"\nDistribución REAL  → media: {REAL_MEAN:.2f}, std: {REAL_STD:.2f}")
print(f"Distribución G     → media: {final_samples.mean():.2f}, std: {final_samples.std():.2f}")
print(f"\nEquilibrio de Nash → D(real) ≈ 0.5: {history['D_real'][-1]:.3f}")
print(f"                   → D(G(z)) ≈ 0.5: {history['D_fake'][-1]:.3f}")
