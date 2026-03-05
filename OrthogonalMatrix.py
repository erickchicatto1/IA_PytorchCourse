import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# Clase para matrices ortogonales
# --------------------------
class OrthogonalMatrix:
    def __init__(self, matrix):
        """
        Inicializa la clase con una matriz cuadrada.
        """
        self.matrix = matrix
        self.n = matrix.shape[0]

    def check_orthonormal_columns(self):
        """
        Verifica que las columnas sean ortonormales.
        """
        cols = [self.matrix[:, i] for i in range(self.n)]
        # Verificar producto punto entre columnas
        for i in range(self.n):
            for j in range(i+1, self.n):
                dot = torch.dot(cols[i], cols[j])
                print(f"Dot product col{i+1}·col{j+1}: {dot:.4f}")
        # Verificar normas
        for i, col in enumerate(cols):
            norm = torch.norm(col)
            print(f"Norma col{i+1}: {norm:.4f}")

    def is_orthogonal(self):
        """
        Verifica que A^T * A = I
        """
        result = torch.matmul(self.matrix.T, self.matrix)
        print("\nMatriz K^T * K =\n", result)
        if torch.allclose(result, torch.eye(self.n)):
            print("\nLa matriz es ortogonal ✅")
            return True
        else:
            print("\nLa matriz no es ortogonal ❌")
            return False

    def plot_columns(self):
        """
        Grafica las columnas como vectores 3D.
        """
        if self.n != 3:
            print("Solo se puede graficar en 3D matrices 3x3.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = torch.zeros(3)
        colors = ['r', 'g', 'b']
        for i in range(self.n):
            vec = self.matrix[:, i]
            ax.quiver(*origin, *vec, color=colors[i], linewidth=2)
            ax.text(*(vec+0.05), f'col{i+1}', color=colors[i], fontsize=12)
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Vectores columna de la matriz ortogonal')
        plt.show()

# --------------------------
# Crear la matriz K
# --------------------------
K = torch.tensor([
    [2/3, 1/3, 2/3],
    [-2/3, 2/3, 1/3],
    [1/3, 2/3, -2/3]
], dtype=torch.float32)

# --------------------------
# Usar la clase
# --------------------------
orth_matrix = OrthogonalMatrix(K)
orth_matrix.check_orthonormal_columns()
orth_matrix.is_orthogonal()
orth_matrix.plot_columns()
