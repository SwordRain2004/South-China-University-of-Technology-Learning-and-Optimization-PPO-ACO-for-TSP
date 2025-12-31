import numpy as np


def generate_tsp(n_city):
    coords = np.random.rand(n_city, 2)
    return coords


def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist
