import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# === CARGA DE DATOS ===
data = pd.read_csv('cal_housing.csv')
x = data.drop('medianHouseValue', axis=1).values
y = data['medianHouseValue'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

# === FUNCIONES DE ESCALAMIENTO ===
def estandar(data):
    m = np.mean(data, axis=0)
    d = np.std(data, axis=0)
    return (data - m) / d

def robusto(data):
    m = np.median(data, axis=0)
    p1 = np.percentile(data, 25, axis=0)
    p2 = np.percentile(data, 75, axis=0)
    ri = p2 - p1
    return (data - m) / ri

# === 1. Regresión lineal ===
modelo_lineal = LinearRegression()
modelo_lineal.fit(x_train, y_train)
y_pred_1 = modelo_lineal.predict(x_test)
mse1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)

# === 2. Polinomial grado 2 sin escalar ===
poly2 = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly2 = poly2.fit_transform(x_train)
x_test_poly2 = poly2.transform(x_test)

modelo_poly2 = LinearRegression()
modelo_poly2.fit(x_train_poly2, y_train)
y_pred_2 = modelo_poly2.predict(x_test_poly2)
mse2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)

# === 3. Polinomial grado 2 con StandardScaler ===
x_train_poly2_std = estandar(x_train_poly2)
x_test_poly2_std = estandar(x_test_poly2)

modelo_poly2_std = LinearRegression()
modelo_poly2_std.fit(x_train_poly2_std, y_train)
y_pred_3 = modelo_poly2_std.predict(x_test_poly2_std)
mse3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)

# === 4. Polinomial grado 2 con RobustScaler ===
x_train_poly2_rob = robusto(x_train_poly2)
x_test_poly2_rob = robusto(x_test_poly2)

modelo_poly2_rob = LinearRegression()
modelo_poly2_rob.fit(x_train_poly2_rob, y_train)
y_pred_4 = modelo_poly2_rob.predict(x_test_poly2_rob)
mse4 = mean_squared_error(y_test, y_pred_4)
r2_4 = r2_score(y_test, y_pred_4)

# === 5. Polinomial grado 3 sin escalar ===
poly3 = PolynomialFeatures(degree=3, include_bias=False)
x_train_poly3 = poly3.fit_transform(x_train)
x_test_poly3 = poly3.transform(x_test)

modelo_poly3 = LinearRegression()
modelo_poly3.fit(x_train_poly3, y_train)
y_pred_5 = modelo_poly3.predict(x_test_poly3)
mse5 = mean_squared_error(y_test, y_pred_5)
r2_5 = r2_score(y_test, y_pred_5)

# === 6. Polinomial grado 3 con StandardScaler ===
x_train_poly3_std = estandar(x_train_poly3)
x_test_poly3_std = estandar(x_test_poly3)

modelo_poly3_std = LinearRegression()
modelo_poly3_std.fit(x_train_poly3_std, y_train)
y_pred_6 = modelo_poly3_std.predict(x_test_poly3_std)
mse6 = mean_squared_error(y_test, y_pred_6)
r2_6 = r2_score(y_test, y_pred_6)

# === 7. Polinomial grado 3 con RobustScaler ===
x_train_poly3_rob = robusto(x_train_poly3)
x_test_poly3_rob = robusto(x_test_poly3)

modelo_poly3_rob = LinearRegression()
modelo_poly3_rob.fit(x_train_poly3_rob, y_train)
y_pred_7 = modelo_poly3_rob.predict(x_test_poly3_rob)
mse7 = mean_squared_error(y_test, y_pred_7)
r2_7 = r2_score(y_test, y_pred_7)

print(f"\n OLS Lineal ")
print(f"MSE: {mse1:.6f}")
print(f"R2:  {r2_1:.6f}")

print(f"\nOLS P2 (sin escalamiento)")
print(f"MSE: {mse2:.6f}")
print(f"R2:  {r2_2:.6f}")

print(f"\nOLS P2 (StandardScaler)")
print(f"MSE: {mse3:.6f}")
print(f"R2:  {r2_3:.6f}")

print(f"\nOLS P2 (RobustScaler) ")
print(f"MSE: {mse4:.6f}")
print(f"R2:  {r2_4:.6f}")

print(f"\n OLS P3 (sin escalamiento)")
print(f"MSE: {mse5:.6f}")
print(f"R2:  {r2_5:.6f}")

print(f"\nOLS P3 (StandardScaler)")
print(f"MSE: {mse6:.6f}")
print(f"R2:  {r2_6:.6f}")

print(f"\nOLS P3 (RobustScaler")
print(f"MSE: {mse7:.6f}")
print(f"R2:  {r2_7:.6f}")