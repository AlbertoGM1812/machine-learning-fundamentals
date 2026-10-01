# -*- coding: utf-8 -*-
"""
Created on 

@author: M
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

dataframe = pd.read_csv('Clusters_spi_global_rankings.csv', encoding='latin1')


#Obtiene las caracteristicas de interés
X = dataframe[['off', 'def']]

# Silueta

best_sil = 0
best_k = 0

#almacena las métricas
calinski = []
davies = []
sils =[]
n_clusters = []

best_cal = 0
cal_k = 0

best_dav = float('inf')
dav_k = 0


for k in range(2,11):
    plt.figure()
    modelo_prueba = KMeans(n_clusters=k, random_state=42,n_init=10)
    silhouette_visualizer = SilhouetteVisualizer(modelo_prueba, colors='yellowbrick') 
    silhouette_visualizer.fit(X)
    
    score = silhouette_visualizer.silhouette_score_
    calinski_score = calinski_harabasz_score(X, modelo_prueba.labels_)
    calinski.append(calinski_score)
    davies_score = davies_bouldin_score(X, modelo_prueba.labels_)   
    davies.append(davies_score) 
   
    n_clusters.append(k)
    
    if score > best_sil:
       best_sil = score
       best_k = k
    
    if calinski_score > best_cal:
       best_cal = calinski_score
       cal_k = k
       
    if davies_score < best_dav:
       best_dav = davies_score
       dav_k = k   
       
    print(f"Coeficiente de silueta con {k} clusters : {score:.2f}")
    silhouette_visualizer.show()
 

print(f"\nRESULTADO: El mejor coeficiente de silueta fue {best_sil:.2f} con {best_k} clusters\n")


plt.figure()
plt.plot(n_clusters, calinski, marker='o')
plt.title('Índice de Calinski-Harabasz para diferentes valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Calinski-Harabasz Score')
plt.scatter(cal_k, best_cal, color='red', label=f'Máximo ({cal_k}, {best_cal})', zorder=5)
plt.grid(True)
plt.show()

plt.figure()
plt.plot(n_clusters, davies, marker='o')
plt.title('Índice de Davies-Bouldin para diferentes valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Davies-Bouldin Score')
plt.scatter(dav_k, best_dav, color='red', label=f'Mínimo ({dav_k}, {best_dav})', zorder=5)
plt.grid(True)
plt.show()

plt.figure()
elbow_visualizer = KElbowVisualizer(modelo_prueba, k=(2, 11))
elbow_visualizer.fit(X) 
elbow_visualizer.show() 
plt.show()
best_k = elbow_visualizer.elbow_value_
print(f"\nEl mejor valor para k de acuerdo con elbow es: {best_k}")


