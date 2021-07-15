#Author: Everton da Silva
#Date: 2021, July

# Importações de Bibliotecas necessárias
import pandas as pd


#Normaliza dados categóricos dividindo as categorias em colunas
def normalize_categoric_data(dataframe, column_name, prefix, console_log):
    """
    Normaliza dados categóricos dividindo as categorias em colunas
    """
    if console_log: print("normalize_categoric_data(dataframe, column_name, prefix, console_log)")
    
    normalized_data = ""
    normalized_data = pd.get_dummies(dataframe[column_name], prefix="sexo")

    if console_log: print(normalized_data)

    return normalized_data

