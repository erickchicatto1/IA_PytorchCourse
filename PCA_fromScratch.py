import numpy as np
import matplotlib.pyplot as plt

# --- Tu clase PCA ---
class PCA:
    
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self,X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        
        cov = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        self.components = eigenvectors[:, :self.n_components].T
        
    def transform(self,X):
        X = X - self.mean
        return np.dot(X, self.components.T)

# --- 1. Crear datos ---
np.random.seed(42)

# datos con correlación (forma inclinada)
mean = [0, 0]
cov = [[3, 2],
       [2, 2]]

X = np.random.multivariate_normal(mean, cov, 200)

# --- 2. Aplicar PCA ---
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)

# --- 3. Visualización ---

plt.figure(figsize=(12,5))

# 🔹 Gráfica 1: datos originales
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], alpha=0.5)
plt.title("Datos originales")

# dibujar componentes principales
mean = pca.mean
for vector in pca.components:
    plt.arrow(mean[0], mean[1],
              vector[0]*3, vector[1]*3,
              color='red', width=0.1)

plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")

# 🔹 Gráfica 2: datos transformados (1D)
plt.subplot(1,2,2)
plt.scatter(X_pca, np.zeros_like(X_pca), alpha=0.5)
plt.title("Datos después de PCA (1D)")
plt.xlabel("Componente principal")

plt.tight_layout()
plt.show()
