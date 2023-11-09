def kNN(D, labels, K, x):
    """
    D is the list of training vectors
    labels is the list of the labels for D
    K is the number of nearest neighbors
    x is the input whose label we want to find
    """
    # this gives euclidean distance but hopefully faster
    S = np.linalg.norm(D - x, axis=1)
    #for sample in D:
    #    S.append(distance(x, sample))
    #S_indices = S.argsort()
    S_indices = np.argpartition(S, range(K))
    # we only need to sort the first K  vectors
    #print(S_indices[:K])
    S = S[S_indices[:K]]
    #print(S)
    labels = labels[S_indices[:K]]
    #print(labels)
    predicted_label = 0
    for k in range(K):
        predicted_label += labels[k]
    return predicted_label > K /2