#Author: Everton da Silva
#Date: 2021, July

import pandas as pd
from imblearn.over_sampling import SMOTE

dados = pd.read_csv('fertility_Diagnosis.txt')

print(dados.head())

dados_label = dados["Output"]

print(dados_label.value_counts())


#Balanceando as classes
def balance_with_smote(dataset, console_log):
    if console_log: print("balance_with_smote(dataset, console_log)")

dados_atributos = dados.drop(columns=['Output'])
dados_classe = dados["Output"]

balanceador = SMOTE()

dados_atributos_b, dados_classes_b = balanceador.fit_resample(dados_atributos, dados_classe)
print(dados_classes_b.value_counts())

from collections import Counter
contagem_classes = Counter(dados_classes_b)
print(contagem_classes)

dados = dados_atributos_b.join(dados_classes_b, how="left")

print("Frequencia de classes pos balanceamento: ")
print(dados.Output.value_counts())
print(dados.head(10))
print(dados.tail(10))
print(dados.shape)

#Criar modelo normalizador
from sklearn.model_selection import train_test_split

atributos = dados.drop(columns="Output")
classes = dados["Output"]

x_train, x_test, y_train, y_test = train_test_split(atributos, classes, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

cfl_forest = RandomForestClassifier()
fertility_RF = cfl_forest.fit(x_train, y_train)
teste_fertility = fertility_RF.predict(x_test)
print("Resultado do Pre-teste do modelo Randon Forest")
print(teste_fertility)
#cfl_forest.predict(x_test, y_test)

for i in range(0,len(x_test)):
    print("Classe: ", y_train.iloc[i], ":" ,teste_fertility[i])

import using_kmeans as us

us.save_model_with_picle(fertility_RF, "normalizador_randon_forest")

#modelo = us.load_model_with_picle("", "normalizador_randon_forest")

cfl_tree = DecisionTreeClassifier()
fertility_DT = cfl_tree.fit(x_train, y_train)
test_fertility_tree = cfl_tree.predict(x_test)
print("Resultado do Pre-teste do modelo Randon Forest")
print(test_fertility_tree)

us.save_model_with_picle(fertility_DT, "normalizador_decision_tree")

#Teste Prelimiar de Acurácia
from sklearn import metrics

print(metrics.accuracy_score(teste_fertility, y_test))
print("RF - Esta Acuracia é precária porque o teste realizado aqui é parcial")

print(metrics.accuracy_score(test_fertility_tree, y_test))
print("DT - Esta Acuracia é precária porque o teste realizado aqui é parcial")