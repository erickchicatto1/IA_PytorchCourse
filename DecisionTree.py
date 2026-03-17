import numpy as np
from collections import Counter

def entropy(y):
    #Paso 1 : contar clases
    hist = np.bincount(y)
    
    #Paso 2 : calcular probabilidades
    total = len(y)
    ps = hist/total
    
    #Paso 3: calcular entropia
    entropy_value = 0
    
    for p in ps:
        if p>0:
            log_p = np.log2(p)
            term = p*log_p
            entropy_value += term
    
    entropy_value = - entropy_value

    return entropy_value


class Node:
    
    def __init__(self,feature=None,threshold =None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold   
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:
    
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
    def fit(self, X, y):
        n_total_features = X.shape[1]

        # Si el usuario no especificó número de features
        if self.n_feats is None:
            self.n_feats = n_total_features

        # Si lo especificó, no puede ser mayor que las disponibles
        elif self.n_feats > n_total_features:
            self.n_feats = n_total_features

        self.root = self._grow_tree(X, y)  
        
    def predict(self,X):
        predictions = []
        
        for x in X:
            pred = self._traverse_tree(x,self.root)
            predictions.append(pred)
        
        return np.array(predictions)  
    
    
    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features = X.shape #Extrae cuántas filas (muestras) y columnas (características) tenemos en este nodo actual.
        n_labels = len(np.unique(y)) #Cuenta cuántas clases distintas quedan. Si solo queda una clase (ej. todos son "Cáncer" o todos son "Sano"), ya no hay necesidad de dividir.
        
        #Criterios de parada
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        #seleccion de caracteristicas
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        #busqueda del mejor corte 
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        #This part is a recursive part 
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    
    
    def _most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x, node.right) 
    
    def _best_criteria(self, X, y, feat_idxs):
            best_gain = -1
            split_idx, split_thresh = None, None
            for feat_idx in feat_idxs:
                X_column = X[:, feat_idx]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    gain = self._information_gain(y, X_column, threshold)

                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = threshold

            return split_idx, split_thresh
            

    def _split(self,X_column,split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
        
        
    
    def _information_gain(self,y,X_column,split_thresh):
        parent_entropy = entropy(y)
        
        #generate split
        left_idxs , right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig
        
        
        
    
    
    
    
    
