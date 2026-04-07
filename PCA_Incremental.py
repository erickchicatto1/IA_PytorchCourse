from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA
import numpy as np
import os

class IncrementalPCADemo:
    def __init__(self, n_components=7, batch_size=200, expand_times=8):
        self.n_components = n_components
        self.batch_size = batch_size
        self.expand_times = expand_times
        self.transformer = IncrementalPCA(
            n_components=self.n_components,
            batch_size=self.batch_size
        )

        self._limit_threads()

    def _limit_threads(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def load_data(self):
        X, _ = load_digits(return_X_y=True)
        return X

    def expand_dataset(self, X):
        for _ in range(self.expand_times):
            X = np.vstack((X, X))
        return X

    def partial_train(self, X):
        self.transformer.partial_fit(X[:100, :])

    def full_train_transform(self, X):
        return self.transformer.fit_transform(X)

    def run(self):
        # 1. Cargar datos
        X = self.load_data()

        # 2. Expandir dataset
        X = self.expand_dataset(X)
        print("Shape original expandido:", X.shape)

        # 3. Entrenamiento parcial
        self.partial_train(X)

        # 4. Entrenamiento completo + transformación
        X_transformed = self.full_train_transform(X)

        # 5. Resultado final
        print("Shape transformado:", X_transformed.shape)

        return X_transformed


# Uso de la clase
if __name__ == "__main__":
    demo = IncrementalPCADemo(n_components=7, batch_size=200, expand_times=8)
    X_reducido = demo.run()
