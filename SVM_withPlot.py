import numpy as np
import matplotlib.pyplot as plt

class SVM:
    
    def __init__(self,learning_rate=0.001,lambda_param=0.01,n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self,X,y):
        n_samples , n_features = X.shape
        y_=np.where(y<=0,-1,1)
        self.w = np.zeros(n_features)
        self.b = 0 
        
        for i in range(self.n_iters):
            for idx , x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i,self.w)-self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self,X):
        approx = np.dot(X,self.w)-self.b
        return np.sign(approx)
    
def plot_svm(X, y, model):
    def get_hyperplane_value(x, w, b, offset):
        return ( -w[0] * x + b + offset ) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # scatter de puntos
    ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')

    # rango de valores
    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:,0])

    # frontera de decisión (w·x - b = 0)
    x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
    x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

    # márgenes (w·x - b = ±1)
    x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, 1)
    x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, 1)

    x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, -1)
    x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, -1)

    # dibujar líneas
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")   # línea central
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k") # margen +
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k") # margen -

    plt.show()
                
