#Normalizar Dados
#Author: Everton da Silva
#Date: 2021 July, 13

# 1 - Normalizar uma base
# 2 - Salvar o modelo de normalização
# 3 - Normalizar novas instancias

import pandas as pd

dataframe = pd.read_csv("dados_normalizar.csv", sep=";")

dados_num = dataframe.drop(columns=["sexo"])

dados_cat = dataframe["sexo"]

print(dados_num.head())

print(dados_cat.head())

#Converter os dados numericos em uma matriz numerica (nd_array)
dados_num = dados_num.values
print(type(dados_num))

#normalizar os numeros
#A. Manualmente com MinMax
# Z =   X - min(dados)
#       ________________
#    max(dados) - min(dados)

dados_num_normalizados = (dados_num-dados_num.min())/(dados_num.max()-dados_num.min())
print(dados_num_normalizados)