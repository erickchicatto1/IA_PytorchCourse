import numpy as np 
from collections import Counter 

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)


class KNN:
    def __init__(self,k=3):
        self.k = k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y 
    
    def predict(self,X):
        predictions = []
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)
    
    def _predict(self,x):
        
        #1. Calculas las distancias 
        distances = []
        for x_train in self.X_train:
            d = euclidean_distance(x,x_train)
            distances.append(d)
            
        #2. obtener indices de los k mas cercanos
        sorted_indices = np.argsort(distances)
        k_idx = sorted_indices[:self.k]
        
        #3. Obtener etiquetas de esos vecinos 
        k_neighbor_labels = []
        for i in k_idx:
            k_neighbor_labels.append(self.y_train[i])
        
        #4. Encontrar la clase mas comun 
        counter = Counter(k_neighbor_labels)
        most_common = counter.most_common(1)
            
        return most_common[0][0]
        
        
        
        
        
        
        
    
