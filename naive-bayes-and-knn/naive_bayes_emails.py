import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paso 1: Cargar el dataset
# Asegúrate que el archivo emails.csv esté en el mismo directorio
df = pd.read_csv("emails.csv")

# Paso 2: Separar características y etiquetas
# Suponemos que:
# - La columna 0 es 'Email No.' (ID) → se elimina
# - La última columna es 'Prediction' → variable objetivo
X = df.iloc[:, 1:-1].values  # Todas menos primera y última columna
y = df.iloc[:, -1].values    # Columna Prediction

# Paso 3: Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

# Paso 4: Validación cruzada k=5
kf = KFold(n_splits=5)

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

# Prueba final con GaussianNB
gaussian_model.fit(X_train, y_train)
y_pred_test_g = gaussian_model.predict(X_test)
acc_test_g = accuracy_score(y_test, y_pred_test_g)
print(f"\n=== Prueba final - GaussianNB ===")
print(f"Distribución: Normal")
print(f"Accuracy en prueba: {acc_test_g:.4f}")

# MultinomialNB (para conteos de palabras)
multinomial_model = MultinomialNB()
acc_multinomial, avg_multinomial = evaluar_modelo(multinomial_model, X_train, y_train, "MultinomialNB")

# Prueba final con MultinomialNB
multinomial_model.fit(X_train, y_train)
y_pred_test_m = multinomial_model.predict(X_test)
acc_test_m = accuracy_score(y_test, y_pred_test_m)
print(f"\n=== Prueba final - MultinomialNB ===")
print(f"Distribución: Multinomial")
print(f"Accuracy en prueba: {acc_test_m:.4f}")


print("\nMatriz de confusión - GaussianNB")
print(confusion_matrix(y_test, y_pred_test_g))
print("Reporte de clasificación - GaussianNB")
print(classification_report(y_test, y_pred_test_g))

