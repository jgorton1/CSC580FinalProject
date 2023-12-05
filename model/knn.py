import numpy as np
def kNN(D, labels, K, x):
    """
    D is the list of training vectors
    labels is the list of the labels for D
    K is the number of nearest neighbors
    x is the input whose label we want to find
    """
    #print(D.shape)
    #print(labels.shape)
    #print(x.shape)
    #assert(False)
    # this gives euclidean distance but hopefully faster
    S = np.linalg.norm(D - x, axis=1)
    S_indices = np.argpartition(S, range(K))
    # we only need to sort the first K  vectors
    #print(S_indices[:K])
    S = S[S_indices[:K]]
    #print(S)
    labels = labels[S_indices[:K]]
    # plurality vote
    predicted_label = np.argmax(np.bincount(labels))
    return predicted_label
def testErrorKNN(tr_values, tr_labels, K, num_tr_values, test_values, test_labels):
    """
    This tests error for the knn algorithm
    (tr/test)_values are the feature vectors, labels are the corresponding labels
    K is number of nearest neighbors
    num_tr_values is number of values from the training set that should be used to train
    Everything but K (int) is a numpy array
    
    """
    tr_values = tr_values[:num_tr_values]
    tr_labels = tr_labels[:num_tr_values]
    # since true == 1, this works for binary classification
    error = sum([kNN(tr_values, tr_labels, K, test_values[i]) != test_labels[i] \
                 for i in range(test_values.shape[0])])
    return error / test_values.shape[0]