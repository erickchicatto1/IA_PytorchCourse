import numpy as np

# Definir vectores
a = np.array([3, 4])
b = np.array([1, 2])

# Producto punto
dot_product = np.dot(a, b)

# Magnitud de b al cuadrado
b_magnitude_squared = np.dot(b, b)

# Proyección de a sobre b
projection = (dot_product / b_magnitude_squared) * b

print("Vector a:", a)
print("Vector b:", b)
print("Proyección de a sobre b:", projection)
