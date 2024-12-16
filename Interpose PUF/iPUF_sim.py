from array import array
from pypuf.simulation import InterposePUF
from pypuf.metrics import accuracy, similarity
from pypuf.io import ChallengeResponseSet, random_inputs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Parametry eksperymentu
Num_of_challenges = 1000
seed = 1
size_tab = np.array([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], dtype='int')
noisiness_tab = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype='float')

# Macierz do przechowywania wyników
Z = np.empty(shape=[len(size_tab), len(noisiness_tab)], dtype='float')

# Trzy konkretne przypadki do testowania
test_cases = [(4, 4), (6, 6), (8, 8)]

results = []

for k_up, k_down in test_cases:
    # Pętla po różnych rozmiarach PUF
    for size in range(len(size_tab)):
        # Pętla po różnych wartościach szumu
        for noise in range(len(noisiness_tab)):
            print("Rozmiar IPUF:", size_tab[size], "Noisiness:", noisiness_tab[noise], "k_up:", k_up, "k_down:", k_down)
            
            # Tworzenie dwóch instancji Interpose PUF z różnymi seedami
            puf = InterposePUF(n=size_tab[size], k_up=k_up, k_down=k_down, seed=seed, noisiness=noisiness_tab[noise])
            puf2 = InterposePUF(n=size_tab[size], k_up=k_up, k_down=k_down, seed=seed+1, noisiness=noisiness_tab[noise])
            
            # Generowanie losowych wyzwań
            r = puf.eval(random_inputs(n=size_tab[size], N=Num_of_challenges, seed=seed))
            
            # Tworzenie zestawu testowego
            test_set = ChallengeResponseSet.from_simulation(puf, N=Num_of_challenges, seed=seed)
            
            # Generowanie histogramu odpowiedzi
            hist, bin_edges = np.histogram(r, bins=2)
            print(hist)
            
            # Obliczanie dokładności (accuracy) i podobieństwa (similarity)
            acc = accuracy(puf, test_set)
            puf_distance = similarity(puf, puf2, seed=31415)
            print("accuracy:", acc, "distance:", puf_distance, "size:", size_tab[size], "Noisiness:", noisiness_tab[noise])
            Z[size][noise] = puf_distance

            print("---------------------------------------")

    # Tworzenie wykresu 3D
    X, Y = np.meshgrid(size_tab, noisiness_tab, indexing='ij')

    wykres = matplotlib.pyplot.figure()
    w = wykres.add_subplot(projection='3d')
    w.set_proj_type('ortho')
    w.view_init(elev=30, azim=45)

    # Rysowanie powierzchni
    w.plot_surface(X, Y, Z, cmap=plt.cm.Blues, linewidth=.5, rstride=1, cstride=1)

    # Dodawanie do wyników
    results.append((k_up, k_down, Z.copy()))

    # Zapis wykresu do pliku
    plt.savefig(f'iPUF_dist_kup{k_up}_kdown{k_down}.png', dpi=300)

# Zwrócenie trzech przypadków
for i, result in enumerate(results[:3]):
    k_up, k_down, Z = result
    print(f"Przypadek {i+1}: k_up={k_up}, k_down={k_down}")
    X, Y = np.meshgrid(size_tab, noisiness_tab, indexing='ij')

    plt.figure()
    ax = plt.subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.Blues, linewidth=.5, rstride=1, cstride=1)
    plt.title(f'Interpose PUF - k_up={k_up}, k_down={k_down}')
    plt.xlabel('PUF Size')
    plt.ylabel('Noisiness')
    ax.set_zlabel('Distance')
    plt.show()
