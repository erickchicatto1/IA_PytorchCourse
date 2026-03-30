import numpy as np

class KalmanFilter:
    def __init__(self, dt):
        # Paso de tiempo
        self.dt = dt
        
        # Estado inicial [posición, velocidad]
        self.x = np.array([[0],
                           [0]])
        
        # Matriz de transición
        self.F = np.array([[1, dt],
                           [0, 1]])
        
        # Matriz de observación (solo medimos posición)
        self.H = np.array([[1, 0]])
        
        # Covarianza inicial
        self.P = np.eye(2)
        
        # Ruido del proceso
        self.Q = np.array([[1e-4, 0],
                           [0, 1e-4]])
        
        # Ruido de medición
        self.R = np.array([[1]])
    
    def predict(self):
        # Predicción del estado
        self.x = np.dot(self.F, self.x)
        
        # Predicción de la covarianza
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x
    
    def update(self, z):
        # Innovación
        y = z - np.dot(self.H, self.x)
        
        # Covarianza de la innovación
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # Ganancia de Kalman
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Actualización del estado
        self.x = self.x + np.dot(K, y)
        
        # Actualización de la covarianza
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x
