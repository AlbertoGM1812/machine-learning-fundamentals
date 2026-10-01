import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tabulate import tabulate

def metricas(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return [accuracy, precision, recall, f1]

def prod_punto(alpha, c):
    return np.dot(alpha, c)

def proyeccion(alpha, c, mag_c):
    return np.dot(alpha, c) / mag_c

def pertenece_a(proy, magnitud, rango):
    if proy >= magnitud and rango[0] > magnitud:
        return 1
    elif proy < magnitud and rango[0] < magnitud:
        return 1
    elif rango[0] > magnitud and proy >= magnitud:
        return 1
    elif rango[0] < magnitud and proy < magnitud:
        return 1
    return 0

def sum_vectores(vec1, vec2):
    return (np.array(vec1) + np.array(vec2)) / 2

def mag_vectores(vec):
    return np.linalg.norm(vec)

def centro_s(grupo, grupo_y, etiqueta):
    grupo = np.array(grupo)
    grupo_y = np.array(grupo_y)
    grupo_p = grupo[grupo_y == etiqueta]
    grupo_np = grupo[grupo_y != etiqueta]
    centro = np.mean(grupo_p, axis=0)
    centro_ng = np.mean(grupo_np, axis=0)
    return [centro, centro_ng]

def one_vs_all(X, y, target_u):
    lista_N = [0 for _ in range(len(target_u))]
    lista_grupos_y = [[] for _ in range(len(target_u))]
    lista_grupos_x = [[] for _ in range(len(target_u))]

    for j in range(len(target_u)):
        count = 0
        for i in range(len(X)):
            if y[i] == target_u[j]:
                lista_grupos_y[j].append(y[i])
                lista_grupos_x[j].append(X[i])
                count += 1
            else:
                lista_grupos_y[j].append(-1)
                lista_grupos_x[j].append(X[i])
        lista_N[j] = count

    return lista_grupos_x, lista_grupos_y, lista_N

def svm(grupo_x, grupo_y, target_unicos, lista_n):
    lista_vectores_finales = [0 for _ in range(len(target_unicos))]
    lista_magnitudes_finales = [0 for _ in range(len(target_unicos))]
    lista_centros = [0 for _ in range(len(target_unicos))]
    lista_centros_magnitudes_finales = []

    for i in range(len(grupo_x)):
        c = centro_s(grupo_x[i], grupo_y[i], target_unicos[i])
        lista_centros[i] = c

    for i in range(len(lista_centros)):
        lista_vectores_finales[i] = sum_vectores(lista_centros[i][0], lista_centros[i][1])
        lista_magnitudes_finales[i] = mag_vectores(lista_vectores_finales[i])
        m1 = mag_vectores(lista_centros[i][0])
        m2 = mag_vectores(lista_centros[i][1])
        lista_centros_magnitudes_finales.append([m1, m2])

    return lista_vectores_finales, lista_magnitudes_finales, lista_centros_magnitudes_finales

def evaluar_svm(x_test, lista_vectores_finales, lista_magnitudes_finales, lista_centros_magnitudes_finales, lista_n, N, lista_labels, target_names):
    lista_predicciones = []
    print("3. Etapa de pruebas:\n")
    for i in range(len(x_test)):
        print(f"Prueba {i+1}:")
        tabla_datos = []
        lista_prob_f = []
        lista_proy = []

        for j in range(len(lista_vectores_finales)):
            proy = proyeccion(x_test[i], lista_vectores_finales[j], lista_magnitudes_finales[j])
            prob = pertenece_a(proy, lista_magnitudes_finales[j], lista_centros_magnitudes_finales[j])
            prob_final = prob * (lista_n[j] / N)
            lista_prob_f.append(prob_final)
            lista_proy.append(proy)
            tabla_datos.append([
                f"Clase {lista_labels[j]}",
                f"{proy:.4f}",
                f"{prob_final:.4f}"
            ])

        print(tabulate(tabla_datos, headers=["Clase", "Proyección", "Probabilidad"], tablefmt="fancy_grid"))

        pred = lista_prob_f.index(max(lista_prob_f))
        print(f"→ Predicción: Clase {pred} ({target_names[lista_labels[pred]]})")
        print(f"  Probabilidad más alta: {max(lista_prob_f):.4f}")
        print(f"  Proyección correspondiente: {lista_proy[pred]:.4f}\n")
        lista_predicciones.append(pred)
    return lista_predicciones

# --------------------
# BLOQUE PRINCIPAL
# --------------------
if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_unicos = list(set(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # ONE VS ALL
    grupo_x, grupo_y, lista_n = one_vs_all(X_train, y_train, target_unicos)
    N = sum(lista_n)

    print("1. Datasets One vs All:")
    for i in range(len(grupo_x)):
        print(f"\nDataset {i+1} (Clase {target_unicos[i]}):")
        print(f" N_{i+1} = {lista_n[i]} (positivos reales)")
    print()

    # ENTRENAMIENTO
    vectores_f, magnitudes_f, centros_mags = svm(grupo_x, grupo_y, target_unicos, lista_n)
    print("2. Etapa de entrenamiento:")
    for i, etiqueta in enumerate(target_unicos):
        print(f"◦ N{i+1}: {lista_n[i]}")
        print(f"  Magnitudes de los centros: {[f'{v:.4f}' for v in centros_mags[i]]}")
        print(f"  Magnitud del vector final: {magnitudes_f[i]:.4f}\n")

    # PRUEBA
    predicciones = evaluar_svm(X_test, vectores_f, magnitudes_f, centros_mags, lista_n, N, target_unicos, iris.target_names)

    # REPORTE FINAL
    print("4. Reporte de clasificación:")
    m = metricas(y_test, predicciones)
    print(f"Accuracy:  {m[0]:.4f}")
    print(f"Precisión: {m[1]:.4f}")
    print(f"Recall:    {m[2]:.4f}")
    print(f"F1 Score:  {m[3]:.4f}\n")

    print(classification_report(y_test, predicciones, digits=2))

    print("5. Matriz de Confusión:")
    matriz = confusion_matrix(y_test, predicciones)
    print(tabulate(matriz, headers=[f"Pred {i}" for i in range(len(matriz))], showindex=[f"Real {i}" for i in range(len(matriz))], tablefmt="fancy_grid"))
