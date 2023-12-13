import pandas
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

dataset = pandas.read_csv("results_no_zeros.csv")
dataset['cluster'] = 0

### k mean clustering ###########################################

# def run_kmeans(n, dataset):
# 	machine = KMeans(n_clusters=n)
# 	machine.fit(dataset)
# 	results = machine.predict(dataset)
# 	centroids = machine.cluster_centers_
# 	pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
# 	pyplot.scatter(centroids[:,0], centroids[:,1],c="red", marker = "*", s = 200)
# 	pyplot.savefig("scatterplot_kmeans_" + str(n) + ".png")
# 	pyplot.close()
# 	return silhouette_score(dataset, results, metric = "euclidean")

# n_list = [2,3,4,5,6,7,8,9,10]
# silhouette_score_list = [run_kmeans(i, dataset) for i in n_list]

# pyplot.scatter(n_list, silhouette_score_list)
# pyplot.savefig("silhouette_score_kmm.png")
# pyplot.close()

## Since the silhouette score for 4 clusters was highest ~.5, we will go with 4
def run_kmeans(n, dataset):
	machine = KMeans(n_clusters=n)
	machine.fit(dataset)
	dataset['cluster'] = machine.predict(dataset)

run_kmeans(4,dataset)

## add clusters to the dataset as cluster IDs and save clustered data
print(dataset)
dataset.to_csv("clustered_data.csv", index=False)

# ### kMedoids clustering ###########################################

# def run_kmedoids(n, dataset):
# 	machine = KMedoids(n_clusters=n)
# 	machine.fit(dataset)
# 	results = machine.predict(dataset)
# 	#centroids = machine.cluster_centers_
# 	# pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
# 	# pyplot.scatter(centroids[:,0], centroids[:,1],c="red", marker = "*", s = 200)
# 	# pyplot.savefig("scatterplot_kmedoids_" + str(n) + ".png")
# 	# pyplot.close()
# 	return silhouette_score(dataset, results, metric = "euclidean")

# n_list = [2,3,4,5,6,7,8,9,10]
# silhouette_score_list = [run_kmedoids(i, dataset) for i in n_list]

# pyplot.scatter(n_list, silhouette_score_list)
# pyplot.savefig("silhouette_score_medoids.png")
# pyplot.close()

# ### Gaussian Mixture #########################################

# def run_gmm(n, dataset):
#   machine = GaussianMixture(n_components=n)
#   machine.fit(dataset)
#   results = machine.predict(dataset)
#   centroids = machine.means_
#   pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
#   pyplot.scatter(centroids[:,0], centroids[:, 1], c="red", marker="*", s=300)
#   pyplot.savefig("scatterplot_gmm_" + str(n) + ".png")
#   pyplot.close()
#   return silhouette_score(dataset, results, metric="euclidean")

# n_list = [2,3,4,5,6,7,8,9,10]
# silhouette_score_list = [run_gmm(i, dataset) for i in n_list]

# pyplot.scatter(n_list, silhouette_score_list)
# pyplot.savefig("silhouette_score_gmm.png")
# pyplot.close()








