import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward

xy = pickle.load(open('../values.pickle'))

plt.figure()
dendrogram(ward(xy))
plt.show()
while True:
    pass
