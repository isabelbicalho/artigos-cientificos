import pylab
import json

datasets = {
    'noisy_circles': 'Noisy circles',
    'noisy_moons': 'Noisy moons',
    'blobs': 'Blobs',
    'no_structure': 'No structure',
    'aniso': 'Aniso',
    'varied': 'Varied'
}

result = json.load(open('out3', 'r'))
x = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

for key in datasets.keys():
    pylab.close()
    pylab.title(datasets[key]+' dataset performance')
    pylab.xlabel('number of samples')
    pylab.ylabel('time (s)')
    pylab.plot(x, result[key]['hierarchical'], label='Hierarchical')
    pylab.plot(x, result[key]['kmeans'], c='m', label='K-Means')
    pylab.plot(x, result[key]['dbscan'], c='g', label='DBSCAN')
    pylab.legend(loc='upper right')
    pylab.savefig('performance_'+key+'.png')
