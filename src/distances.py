import numpy as np

def euclidean_distances(X, Y):
    distances=np.empty([])
    for x in X:
        for y in Y:
            distances=np.append(distances, np.linalg.norm(x-y))
    distances=np.delete(distances,0)
    distances=np.reshape(distances, (X.shape[0], Y.shape[0]))
    return distances

def manhattan_distances(X, Y):
    distances=np.empty([])
    for x in X:
        for y in Y:
            distances=np.append(distances, np.sum(abs(x-y)))
    distances=np.delete(distances,0)
    distances=np.reshape(distances, (X.shape[0], Y.shape[0]))
    return distances


def cosine_distances(X, Y):
    distances=np.empty([])
    for x in X:
        for y in Y:
            distances=np.append(distances, 1 - ( np.inner(x, y) / (np.sqrt(np.inner(x,x)) * np.sqrt(np.inner(y,y)))))
    distances=np.delete(distances,0)
    distances=np.reshape(distances, (X.shape[0], Y.shape[0]))
    return distances
