from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar datos
data = load_iris()
X = data.data
y = data.target

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear modelo XGBoost
model = XGBClassifier(
    n_estimators=100,   # número de árboles
    learning_rate=0.1,  # qué tan rápido aprende
    max_depth=3         # profundidad de árboles
)

# 4. Entrenar modelo
model.fit(X_train, y_train)

# 5. Hacer predicciones
y_pred = model.predict(X_test)

# 6. Evaluar
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
