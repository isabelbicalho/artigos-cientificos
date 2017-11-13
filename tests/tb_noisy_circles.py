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
while n_samples <= 3000:
    dataset = pickle.load(open('data/datasets_'+str(n_samples)+'.pyc','r'))
    for key in dataset.keys():
        kmeans = KMeans(n_clusters=params[key]['n_clusters']) 
        hierarchical = AgglomerativeClustering(n_clusters=params[key]['n_clusters']) 
        dbscan = DBSCAN(eps=params[key]['eps'])
        data = dataset[key][0]
        init_kmeans = time.time()
        kmeans.fit(data)
        end_kmeans = time.time()
        init_hier = time.time()
        hierarchical.fit(data)
        end_hier = time.time()
        init_dbscan = time.time()
        dbscan.fit(data)
        end_dbscan = time.time()
        execution_time[key]['kmeans'].append(end_kmeans - init_kmeans)
        execution_time[key]['hierarchical'].append(end_hier - init_hier)
        execution_time[key]['dbscan'].append(end_dbscan - init_dbscan)

    n_samples += 250

import json
print json.dumps(execution_time, indent=4, sort_keys=True)
