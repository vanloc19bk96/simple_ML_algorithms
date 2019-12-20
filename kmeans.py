import numpy as np

def fit(X, k, max_iters=300, tol=1e-4):
    centroids = {}
    classes = {}

    # init centroids
    for i in range(k):
        centroids[i] = X[i]
    for i in range(max_iters):
        for i in range(k):
            classes[i] = []
        for x in X:
            distances = [np.linalg.norm(x - centroids[centroid]) for centroid in centroids]
            nearest = np.argmin(distances)
            classes[nearest].append(x)

        previous = dict(centroids)
        for c in classes:
            centroids[c] = np.average(classes[c], axis=0)

        isOptimal = False
        for centroid in centroids:
            original_centroid = previous[centroid]
            current_centroid = centroids[centroid]
            if sum((current_centroid - original_centroid) / original_centroid) * 100 > tol:
                isOptimal = True
        if isOptimal:
            break
    return centroids
