import numpy as np
from collections import Counter

# --- Función de entropía ---
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    e = 0
    for p in ps:
        if p > 0:
            e += p * np.log2(p)
    return -e

# --- Clases Node y DecisionTree ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=3, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        n_total_features = X.shape[1]
        self.n_feats = n_total_features if self.n_feats is None else min(self.n_feats, n_total_features)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
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

    def _split(self, X_column, split_thresh):
        left = np.argwhere(X_column <= split_thresh).flatten()
        right = np.argwhere(X_column > split_thresh).flatten()
        return left, right

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

# --- Función para imprimir árbol en consola ---
def print_tree(node, feature_names=None, depth=0):
    indent = "  " * depth
    if node.is_leaf_node():
        print(indent + f"Leaf: Clase={node.value}")
    else:
        feature = node.feature if feature_names is None else feature_names[node.feature]
        print(indent + f"[{feature} <= {node.threshold:.3f}]")
        print(indent + "-> Izquierda:")
        print_tree(node.left, feature_names, depth + 1)
        print(indent + "-> Derecha:")
        print_tree(node.right, feature_names, depth + 1)

# --- Función para visualizar con graphviz ---
from graphviz import Digraph

def add_nodes_edges(node, dot=None, node_id=0):
    if dot is None:
        dot = Digraph()
    if node.is_leaf_node():
        dot.node(name=str(node_id), label=f"Clase = {node.value}", shape='box', style='filled', color='lightgrey')
    else:
        label = f"X[{node.feature}] <= {node.threshold:.3f}"
        dot.node(name=str(node_id), label=label)
        left_id = node_id * 2 + 1
        right_id = node_id * 2 + 2
        dot.edge(str(node_id), str(left_id), label="True")
        dot.edge(str(node_id), str(right_id), label="False")
        add_nodes_edges(node.left, dot, left_id)
        add_nodes_edges(node.right, dot, right_id)
    return dot

# --- Ejemplo de uso con dataset simple ---
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# Imprimir árbol en texto
feature_names = ["Feature 0", "Feature 1"]
print_tree(tree.root, feature_names)

# Visualizar árbol con graphviz (asegúrate de tener instalado graphviz y el software)
dot = add_nodes_edges(tree.root)
dot.render('decision_tree', format='png', cleanup=True)
dot.view()
