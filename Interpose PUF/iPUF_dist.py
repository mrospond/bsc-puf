from array import array
from pypuf.simulation import InterposePUF
from pypuf.metrics import accuracy, similarity
from pypuf.io import ChallengeResponseSet, random_inputs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Num_of_challenges = 1000
seed = 1
size_tab = np.array([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], dtype='int')
noisiness_tab = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype='float')

Z = np.empty(shape=[len(size_tab), len(noisiness_tab)], dtype='float')

for size in range(len(size_tab)):
    for noise in range(len(noisiness_tab)):
        print("Rozmiar IPUF:", size_tab[size], "Noisiness:", noisiness_tab[noise])
        
        puf = InterposePUF(n=size_tab[size], k_up=8, k_down=8, seed=seed, noisiness=noisiness_tab[noise])
        puf2 = InterposePUF(n=size_tab[size], k_up=8, k_down=8, seed=seed+1, noisiness=noisiness_tab[noise])
        r = puf.eval(random_inputs(n=size_tab[size], N=Num_of_challenges, seed=seed))
        test_set = ChallengeResponseSet.from_simulation(puf, N=Num_of_challenges, seed=seed)
        
        hist, bin_edges = np.histogram(r, bins=2)
        print(hist)
       
        acc = accuracy(puf, test_set)
        puf_distance = similarity(puf, puf2, seed=31415)
        print("accuracy:", acc, "distance:", puf_distance, "size:", size_tab[size], "Noisiness:", noisiness_tab[noise])
        Z[size][noise] = puf_distance

        print("---------------------------------------")

X, Y = np.meshgrid(size_tab, noisiness_tab, indexing='ij')

wykres = matplotlib.pyplot.figure()
w = wykres.add_subplot(projection='3d')
w.set_proj_type('ortho')
w.view_init(elev=30, azim=45)

w.plot_surface(X, Y, Z, cmap=plt.cm.Blues, linewidth=.5, rstride=1, cstride=1)

plt.savefig('iPUF_dist.png', dpi=300)
