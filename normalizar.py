#Normalizar Dados
#Author: Everton da Silva
#Date: 2021 July, 13

# 1 - Normalizar uma base
# 2 - Salvar o modelo de normalização
# 3 - Normalizar novas instancias

import pandas as pd
from sklearn import preprocessing

dataframe = pd.read_csv("dados_normalizar.csv", sep=";")
dados_num = dataframe.drop(columns=["sexo"])
dados_cat = dataframe["sexo"]

#print(dados_num.head())

#print(dados_cat.head())

#Converter os dados numericos em uma matriz numerica (nd_array)
dados_num = dados_num.values
#print(type(dados_num))

#normalizar os numeros
#A. Manualmente com MinMax
# Z =   X - min(dados)
#       ________________
#    max(dados) - min(dados)

dados_num_normalizados = (dados_num-dados_num.min())/(dados_num.max()-dados_num.min())
#print(dados_num_normalizados)

#Normalizar utilizando um pacote pronto
normalizador = preprocessing.MinMaxScaler()

#Obter o resultado do modelo normalizador para a base processada
#dados_num_normalizados_2 = normalizador.fit_transform(dados_num)
#print(dados_num_normalizados_2)


#Obter o modelo normalizador para a base processada
modelo_normalizador = normalizador.fit(dados_num)
#print(dados_num_normalizados_2)

# Salvar o modelo normalizador
from pickle import dump
dump(modelo_normalizador, open("modelo_normalizador.pkl", "wb"))

#Carregar o modelo normalizador salvo
from pickle import load
normalizador = load(open("modelo_normalizador.pkl", "rb"))

dados_num_normalizados_2 = normalizador.fit_transform(dados_num)
#print(dados_num_normalizados_2)

#Normalizar os dados Categoricos(Coluna Sexo)
#print("Classes Originais")
dados_cat_normalizados = pd.get_dummies(dados_cat, prefix="sexo")
#print(dados_cat_normalizados)

#Recompor os dados para obter o modelo de Machine Learning
#Converter o ndarray para o dataframe
dados_num = pd.DataFrame(dados_num_normalizados_2, columns=["Idade", "Altura", "Peso"])

#Juntar com as categorias normalizadas
dados_finais = dados_num.join(dados_cat_normalizados, how="left")
#print(dados_finais)