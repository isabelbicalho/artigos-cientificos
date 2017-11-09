from sklearn.datasets       import make_blobs
from sklearn.cluster        import KMeans
#from adspy_shared_utilities import plot_labelled_scatter

x, y = make_blobs(random_state = 1000)
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
#plot_labelled_scatter(x, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
