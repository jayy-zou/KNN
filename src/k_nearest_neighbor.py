import numpy as np
from .distances import euclidean_distances, manhattan_distances, cosine_distances
from scipy import stats

class KNearestNeighbor():
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.targets = None


    def fit(self, features, targets):
        self.features = features
        self.targets = targets

    def predict(self, features, ignore_first=False):
        if self.distance_measure == "euclidean":
            distance_array = euclidean_distances(self.features, features)
        elif self.distance_measure == "manhattan":
            distance_array = manhattan_distances(self.features, features)
        else:
            distance_array = cosine_distances(self.features, features)

        labels = np.zeros((self.features.shape[0], self.targets.shape[1]))
        for row in range(len(features)):
            if (ignore_first):
                index = distance_array[row].argsort()[1:self.n_neighbors + 1]
            else:
                index = distance_array[row].argsort()[:self.n_neighbors]

            targets= self.targets.copy()[index]

            if self.aggregator == "mode":
                prediction = np.zeros((1, np.shape(targets)[1]))
                for i in range(0, targets.shape[1]):
                    column = targets[:, i]
                    not_using, index, counts = np.unique(targets[:, i], axis=0, return_index=True, return_counts=True)
                    prediction[0][i] = column[index[np.argmax(counts)]]
            elif self.aggregator == "mean":
                prediction = np.mean(targets, axis=0)
            else:
                prediction = np.median(targets, axis=0)

            labels[row] = prediction

        return labels
