# -*- coding: utf-8 -*-
"""
Clustering sobre datos de fútbol (off y def)
"""

import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import simpledialog
from sklearn.cluster import KMeans

# Leer CSV
dataframe = pd.read_csv('Clusters_spi_global_rankings.csv', encoding='latin1')

# Características de interés
X = dataframe[['off', 'def']]

# Crear ventana para pedir valor de k
root = tk.Tk()
root.withdraw() 
k = simpledialog.askinteger("Valor de K", "Introduce el valor de K:")

# Si no se introduce, asignar k = 4
if k is None:
    print("\nNo se introdujo ningún número, se asignará k = 4")
    k = 4
else:
    print(f"\nEl número capturado es: {k}")

# Inicializar y entrenar modelo
kmeansModel = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeansModel.fit(X)

# Obtener etiquetas y centroides
dataframe['Class'] = kmeansModel.labels_
centroides = kmeansModel.cluster_centers_

# Graficar en 2D
colors = ['magenta', 'lime', 'cyan', 'red', 'royalblue', 'chocolate', 'gold', 'darkviolet', 'forestgreen', 'gray']

plt.figure(figsize=(8,6))
for i in range(k):
    cluster_data = dataframe[dataframe['Class'] == i]
    plt.scatter(cluster_data['off'], cluster_data['def'], c=colors[i], label=f'Clase {i}', alpha=0.7)

# Graficar centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='x', s=200, label='Centroides')

# Etiquetas
plt.xlabel('Poder ofensivo (off)')
plt.ylabel('Poder defensivo (def)')
plt.title('Clustering K-Means: Equipos de fútbol')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Guardar resultados
dataframe = dataframe.sort_values(by='Class')
dataframe.to_csv('Football_Clusters2.csv', encoding='latin1', index=False)

print("\nClustering completado y guardado como 'Football_Clusters.csv'")
