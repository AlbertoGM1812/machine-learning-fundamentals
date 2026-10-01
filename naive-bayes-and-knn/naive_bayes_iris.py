import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar y mezclar datos
df = pd.read_csv("iris.csv")  # Reemplaza con tu ruta si es necesario
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

# Validación cruzada
kf = KFold(n_splits=5)

# Función para validación cruzada
def evaluar_modelo(modelo, X_data, y_data, nombre=""):
    print(f"\nResultados de validación cruzada - {nombre}")
    accuracies = []
    for i, (train_idx, val_idx) in enumerate(kf.split(X_data), 1):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]
        modelo.fit(X_tr, y_tr)
        y_pred = modelo.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        print(f"Pliegue {i} - Accuracy: {acc:.4f}")
    promedio = np.mean(accuracies)
    print(f"Promedio - Accuracy: {promedio:.4f}")
    return accuracies, promedio

# GaussianNB
gaussian_model = GaussianNB()
acc_gaussian, avg_gaussian = evaluar_modelo(gaussian_model, X_train, y_train, "GaussianNB")

# Entrenamiento y prueba final
gaussian_model.fit(X_train, y_train)
y_pred_test_g = gaussian_model.predict(X_test)
acc_test_g = accuracy_score(y_test, y_pred_test_g)
print(f"\n=== Prueba final - GaussianNB ===")
print(f"Distribución: Normal")
print(f"Accuracy en prueba: {acc_test_g:.4f}")

# MultinomialNB (con valores absolutos para asegurar valores no negativos)
X_train_mnb = np.abs(X_train)
X_test_mnb = np.abs(X_test)
multinomial_model = MultinomialNB()
acc_multinomial, avg_multinomial = evaluar_modelo(multinomial_model, X_train_mnb, y_train, "MultinomialNB")

# Entrenamiento y prueba final
multinomial_model.fit(X_train_mnb, y_train)
y_pred_test_m = multinomial_model.predict(X_test_mnb)
acc_test_m = accuracy_score(y_test, y_pred_test_m)
print(f"\n=== Prueba final - MultinomialNB ===")
print(f"Distribución: Multinomial")
print(f"Accuracy en prueba: {acc_test_m:.4f}")


# === Reporte y matriz para GaussianNB ===
print("\nMatriz de confusión - GaussianNB")
print(confusion_matrix(y_test, y_pred_test_g))
print("Reporte de clasificación - GaussianNB")
print(classification_report(y_test, y_pred_test_g))

