#Author: Everton da Silva
#Date: 2021, July

#1. determinar o numero ótimo de clusters
#2. Obter e Salvar o modelo
#1. Criar um módulo de inferência com os clusters

## Bibliotecas usadas
import pandas as pd
import numpy as np
import math

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
#import seaborn as sbn

from scipy.spatial.distance import cdist

################


#Read Dataset
def read_dataset(console_log):
    data = pd.read_csv("iris.csv")
    if console_log: print(data.head())
    return data

#Converter a coluna do dataset em vetor
def convert_dataframe_to_matrix(dataset, column_to_remove, console_log):
    data = dataset.drop(columns=[column_to_remove])
    data = data.values
    if console_log: print(data)
    return data

#Clusterizar os dados 
def kmeans_model(dados, console_log):
    
    k_model = KMeans(n_clusters=3).fit(dados)
    
    #iris_model = k_model.fit_transform(dados)
        
    #if console_log: print(iris_model)
    if console_log: print(k_model.cluster_centers_)
    #if console_log: print(k_model.cluster_centers_)

    return k_model

def get_distortions(data, model, turns, console_log):
    distortions = []
    for turn in turns:
        k_model = KMeans(n_clusters=turn).fit(data)
        distortions.append(
            sum(np.min(cdist(data, k_model.cluster_centers_, "euclidean"), axis=1)/data.shape[0])
        )
    if console_log: print(distortions)
    return distortions

def plot_and_save_distortions_bow(k):
    fig, ax = plt.subplots()

    ax.plot(k, distorcoes)
    ax.set(xlabel = "n Clusters",
        ylabel = "Distorcao",
        title = "Elbol por Distorcao")
    ax.grid()
    fig.savefig('elbow_distorcao.png')
    plt.show()

def get_intertia(data, model, turns, console_log):
    intertia = []
    for turn in turns:
        k_model = KMeans(n_clusters=turn).fit(data)
        intertia.append(k_model.inertia_)
    if console_log: print(intertia)
    return intertia

def get_distances(sum_square, console_log):
    x1 = 1
    y1 = sum_square[0]

    x2 = 10
    y2 = sum_square[len(sum_square)-1]

    distancias = []
    for i in range(len(sum_square)):
        x0 = i+2
        y0 = sum_square[i]
        numerador = abs((y2-y1) * x0 - (x2 - x1) * y0 + x2 * y1 * x1)
        denominador = math.sqrt((y2 - y1)**2 + (x2-x1)**2)
        distancias.append(numerador/denominador)
    if console_log: print(distancias)
    n_cluster_otimo = distancias.index(max(distancias))+1
    print("Numero otimo de clusters", n_cluster_otimo)
    return n_cluster_otimo

k = range(1,11)

dataframe = read_dataset(False)

data = convert_dataframe_to_matrix(dataframe, "species", False)

model = kmeans_model(data, False)

distorcoes = get_distortions(data, model, k, False)

soma_quadrados = get_intertia(data, model, k, False)

plot_and_save_distortions_bow(k=range(1,11))

get_distances(soma_quadrados, True)

