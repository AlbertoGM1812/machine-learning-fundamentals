import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Función para evaluar KNN
def evaluar_knn(X_train, y_train, X_test, y_test, n_neighbors, weights):
    modelo = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return accuracy_score(y_test, y_pred), y_pred

# Función para imprimir matriz y reporte
def imprimir_matriz_reporte(y_test, y_pred, nombre):
    print(f"\nMatriz de confusión {nombre}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nReporte de clasificación {nombre}:")
    print(classification_report(y_test, y_pred, digits=6, zero_division=0))

# Evaluación general
def evaluar_dataset(nombre, X, y):
    tabla1 = []
    # Mezclar datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, shuffle=True
    )

    # Validación cruzada k=3
    kf = KFold(n_splits=3)
    configs = [(1, 'uniform'), (10, 'uniform'), (10, 'distance')]

    for vecinos, pesos in configs:
        accs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            acc, _ = evaluar_knn(X_tr, y_tr, X_val, y_val, vecinos, pesos)
            accs.append(acc)
            tabla1.append([nombre, vecinos, pesos, fold + 1, round(acc, 6)])
        prom = sum(accs) / len(accs)
        tabla1.append([nombre, vecinos, pesos, '    Promedio', round(prom, 6)])

    # Mejor configuración final (10, distancia)
    acc_knn, y_pred_knn = evaluar_knn(X_train, y_train, X_test, y_test, 10, 'distance')

    # Modelo Bayes
    modelo_bayes = GaussianNB()
    modelo_bayes.fit(X_train, y_train)
    y_pred_bayes = modelo_bayes.predict(X_test)
    acc_bayes = accuracy_score(y_test, y_pred_bayes)

    # Tabla 2
    tabla2 = [
        [nombre, 'Naive Bayes', '-', '-', 'Normal', round(acc_bayes, 6)],
        [nombre, 'K-NN', 10, 'distancia', '-', round(acc_knn, 6)]
    ]

    # Reportes
    imprimir_matriz_reporte(y_test, y_pred_knn, f"KNN ({nombre})")
    imprimir_matriz_reporte(y_test, y_pred_bayes, f"Naive Bayes ({nombre})")

    return tabla1, tabla2

# ===========================
# EJECUCIÓN PARA IRIS
# ===========================
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

tabla1_iris, tabla2_iris = evaluar_dataset("iris.csv", X_iris, y_iris)

# ===========================
# EJECUCIÓN PARA EMAILS
# ===========================
emails = pd.read_csv("emails.csv")
X_emails = emails.iloc[:, 1:-1].values
y_emails = emails.iloc[:, -1].values

tabla1_emails, tabla2_emails = evaluar_dataset("emails.csv", X_emails, y_emails)

# ===========================
# IMPRIMIR TABLAS
# ===========================
print("\n=== TABLA 1: Validación cruzada con k=3 para 1-NN y 10-NN ===")
print("Dataset\tVecinos\tPesos\tPliegue\tAccuracy")
for fila in tabla1_iris + tabla1_emails:
    print("\t".join(str(x) for x in fila))

print("\n=== TABLA 2: Resultados de pruebas finales ===")
print("Dataset\tClasificador\tVecinos\tPesos\tDistribución\tAccuracy")
for fila in tabla2_iris + tabla2_emails:
    print("\t".join(str(x) for x in fila))