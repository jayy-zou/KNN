import numpy as np
from .k_nearest_neighbor import KNearestNeighbor


def collaborative_filtering(input_array, n_neighbors,
                            distance_measure='euclidean', aggregator='mode'):
    model = KNearestNeighbor(n_neighbors, distance_measure, aggregator)
    model.fit(input_array, input_array)
    predictions = model.predict(input_array, True)
    imputed_array = input_array.copy()
    for x in range(np.shape(input_array)[0]):
        for y in range(np.shape(input_array)[1]):
            if input_array[x, y] == 0:
                imputed_array[x, y] = predictions[x, y]

    return imputed_array
