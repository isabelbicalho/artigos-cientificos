import time
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

n_samples = 250
execution_time = {
    'noisy_circles': {
        'kmeans': [],
        'hierarchical': [],
        'dbscan': []
    },
    'noisy_moons': {
        'kmeans': [],
        'hierarchical': [],
        'dbscan': []
    },
    'blobs': {
        'kmeans': [],
        'hierarchical': [],
        'dbscan': []
    },
    'no_structure': {
        'kmeans': [],
        'hierarchical': [],
        'dbscan': []
    },
    'aniso': {
        'kmeans': [],
        'hierarchical': [],
        'dbscan': []
    },
    'varied': {
        'kmeans': [],
        'hierarchical': [],
        'dbscan': []
    }
}

params = {
    'noisy_circles': {
        'eps': .3,
        'n_clusters': 2
    },
    'noisy_moons': {
        'eps': .3,
        'n_clusters': 2
    },
    'blobs': {
        'eps': .3,
        'n_clusters': 3
    },
    'no_structure': {
        'eps': .3,
        'n_clusters': 3
    },
    'aniso': {
        'eps': .15,
        'n_clusters': 3
    },
    'varied': {
        'eps': .18,
        'n_clusters': 3
    }
}

import pylab
dataset = pickle.load(open('data/datasets_1500.pyc','r'))
for key in dataset.keys():
    kmeans = KMeans(n_clusters=params[key]['n_clusters']) 
    hierarchical = AgglomerativeClustering(n_clusters=params[key]['n_clusters']) 
    dbscan = DBSCAN(eps=params[key]['eps'])
    data = dataset[key][0]
    kmeans.fit(data)
    hierarchical.fit(data)
    dbscan.fit(data)
    import pdb; pdb.set_trace()
    colors_dbscan = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in dbscan.labels_])
    colors_kmeans = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in kmeans.labels_])
    colors_hierarchical = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in hierarchical.labels_])
    pylab.close()
    pylab.scatter(data[key][0][:,0], data[key][0][:,1], c=colors_dbscan)
    pylab.savefig('result_dbscan_'+key+'.png')
    pylab.close()
    pylab.scatter(data[key][0][:,0], data[key][0][:,1], c=colors_hierarchical)
    pylab.savefig('result_hierarchical_'+key+'.png')
    pylab.close()
    pylab.scatter(data[key][0][:,0], data[key][0][:,1], c=colors_kmeans)
    pylab.savefig('result_kmeans_'+key+'.png')
