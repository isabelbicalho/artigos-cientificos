from sklearn import datasets
import numpy as np
import pickle
import pylab

np.random.seed(0)

n_samples = 250

while n_samples <= 3000:
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    dataset = {
        'noisy_circles': noisy_circles,
        'noisy_moons': noisy_moons,
        'blobs': blobs,
        'no_structure': no_structure,
        'aniso': aniso,
        'varied': varied
    }
    pickle.dump(dataset, open('datasets_'+str(n_samples)+'.pyc','w'))
    for key in dataset.keys():
        value = dataset[key]
        pylab.close()
        pylab.scatter(value[0][:,0], value[0][:,1])
        pylab.savefig('images/'+key+'_'+str(n_samples)+'.png')
    n_samples += 250
