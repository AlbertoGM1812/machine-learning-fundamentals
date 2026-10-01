# ------------------------------------------------------------
# REGRESION LINEAL Y POLINOMIAL (OLS + SGD)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------------------------------------------
# 1. Cargar datos desde archivo
# ------------------------------------------------------------

df = pd.read_csv("datos.csv")
X = df[["x"]].values
y = df["y"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=0
)


# 2. Solicitar hiperparametros para SGD


def solicitar_entero(msg, min_val=1):
    while True:
        try:
            num = int(input(msg))
            if num >= min_val:
                return num
        except:
            pass
        print(f"Ingrese un entero valido mayor o igual a {min_val}")

def solicitar_float(msg, min_val=1e-10):
    while True:
        try:
            val = float(input(msg))
            if val >= min_val:
                return val
        except:
            pass
        print(f"Ingrese un valor decimal valido mayor o igual a {min_val}")

print("\nConfiguracion del modelo SGD")
iteraciones = solicitar_entero("Numero de iteraciones: ")
alpha = solicitar_float("Tasa de aprendizaje (alpha): ")


# 3. Funcion para graficar resultados


def graficar_modelo(x, y_real, y_pred, titulo, color_datos='green', color_linea='red'):
    orden = np.argsort(x.flatten())
    x_ord = x[orden]
    y_ord = y_pred[orden]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y_real, c=color_datos, edgecolors='black', alpha=0.7, label='Valores reales')
    ax.plot(x_ord, y_ord, color=color_linea, linewidth=2, linestyle='-', label='Modelo ajustado')
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


# 4. Entrenamiento y almacenamiento de resultados

# Lista para almacenar resultados finales
resultados = []

# --- OLS Lineal ---
ols1 = LinearRegression()
ols1.fit(X_train, y_train)
pred1 = ols1.predict(X_test)
graficar_modelo(X_test, y_test, pred1, "OLS Lineal", color_datos="green", color_linea="darkred")
resultados.append(("OLS Lineal", mean_squared_error(y_test, pred1), r2_score(y_test, pred1)))

# --- OLS Grado 2 ---
poly2 = PolynomialFeatures(degree=2)
X_train_g2 = poly2.fit_transform(X_train)
X_test_g2 = poly2.transform(X_test)

ols2 = LinearRegression()
ols2.fit(X_train_g2, y_train)
pred2 = ols2.predict(X_test_g2)
graficar_modelo(X_test, y_test, pred2, "OLS Polinomial G2", color_datos="green", color_linea="darkred")
resultados.append(("OLS Grado 2", mean_squared_error(y_test, pred2),  r2_score(y_test, pred2)))

# --- OLS Grado 3 ---
poly3 = PolynomialFeatures(degree=3)
X_train_g3 = poly3.fit_transform(X_train)
X_test_g3 = poly3.transform(X_test)

ols3 = LinearRegression()
ols3.fit(X_train_g3, y_train)
pred3 = ols3.predict(X_test_g3)
graficar_modelo(X_test, y_test, pred3, "OLS Polinomial G3", color_datos="green", color_linea="darkred")
resultados.append(("OLS Grado 3", mean_squared_error(y_test, pred3), r2_score(y_test, pred3)))



# --- SGD Lineal ---
sgd1 = SGDRegressor(max_iter=iteraciones, alpha=alpha, eta0=alpha,
                    learning_rate='constant', random_state=0, tol=0.00112)
sgd1.fit(X_train, y_train)
pred4 = sgd1.predict(X_test)
graficar_modelo(X_test, y_test, pred4, "SGD Lineal", color_datos="forestgreen", color_linea="red")
resultados.append(("SGD Lineal", mean_squared_error(y_test, pred4), r2_score(y_test, pred4)))

# --- SGD Grado 2 ---
sgd2 = SGDRegressor(max_iter=iteraciones, alpha=alpha, eta0=alpha,
                    learning_rate='constant', random_state=0, tol=0.00112)
sgd2.fit(X_train_g2, y_train)
pred5 = sgd2.predict(X_test_g2)
graficar_modelo(X_test, y_test, pred5, "SGD Polinomial G2", color_datos="forestgreen", color_linea="red")
resultados.append(("SGD Grado 2", mean_squared_error(y_test, pred5), r2_score(y_test, pred5)))

# --- SGD Grado 3 ---
poly_sgd3 = PolynomialFeatures(degree=3)
X_train_g3_sgd = poly_sgd3.fit_transform(X_train)
X_test_g3_sgd = poly_sgd3.fit_transform(X_test)

sgd3 = SGDRegressor(max_iter=iteraciones, alpha=alpha, eta0=alpha,
                    learning_rate='constant', random_state=0, tol=0.00112)
sgd3.fit(X_train_g3_sgd, y_train)
pred6 = sgd3.predict(X_test_g3_sgd)
graficar_modelo(X_test, y_test, pred6, "SGD Polinomial G3", color_datos="forestgreen", color_linea="red")
resultados.append(("SGD Grado 3", mean_squared_error(y_test, pred6), r2_score(y_test, pred6)))

# 6. Impresion de resumen final de todos los modelos

print("\nResumen de resultados (MSE, R2)")
for nombre, mse, r2 in resultados:
    print(f"{nombre:<25} MSE: {mse:.4f}  R2: {r2:.4f}")