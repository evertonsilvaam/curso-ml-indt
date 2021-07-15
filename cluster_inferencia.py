from pickle import load
#from sklearn import sklearn.predict

nova_flor = [[4.9,3.7,1.7,0.1]]

iris_cluster = load(open("cluster_kmeans.pkl", "rb"))
res = iris_cluster.predict(nova_flor)
print(res)

print(iris_cluster.cluster_centers_[res])