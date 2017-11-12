import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle
from sklearn.cluster import DBSCAN

xy = pickle.load(open('../values.pickle'))

cls = DBSCAN(eps=2, min_samples=2)
cls.fit_predict(xy)

