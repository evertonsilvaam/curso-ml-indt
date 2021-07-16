import using_kmeans as us

RF = us.load_model_with_picle("","normalizador_random_forest")
DT = us.load_model_with_picle("","normalizador_decision_tree")

#Teste Prelimiar de Acurácia
from sklearn import metrics

print(metrics.accuracy_score)