import utils.n_fold_cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def backwardSequentialFeatureSelect3Sec(X,y, k, samples_per_song, min_features=1):
    '''
    Selected best feature set for knn based on accuracy
    X - features
    y - labels
    k - k for k-NN
    samples_per_song - dictionary song index -> (number of samples, index of start of song in dataframe)
    '''
    # for 3 second features - we must tts based on song, not each sample
    # for knn
    # with 5-fold cross validation
    
    #X_train, X_val, y_train, y_val = tts_3_sec(normalized_features, numeric_labels, test_size=0.2, random_state=42)
    feat_sel = list(range(X.shape[1]))
    best_feat_sel = feat_sel
    best_score = 0
    while len(feat_sel) > min_features:
        worst_feat = None
        worst_perf = 1
        for feat in feat_sel:
            this_feat = feat_sel.copy()
            this_feat.remove(feat)
            # now do 5-fold cross validation
            accuracy_sum = 0
            for X_train, X_val, y_train, y_val in utils.n_fold_cv.n_fold_cv(X, y, samples_per_song, n=5):
                # Your training and validation process here
                # compute score
                X_train_subset = X_train[:, this_feat]
                X_val_subset = X_val[:, this_feat]
                # use sklearn knn
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_subset, y_train)
                # Make predictions on the test set
                y_pred = knn.predict(X_val_subset)

                # Evaluate the accuracy of the model
                #print(y_val)
                #print(y_pred)
                #print()
                accuracy = accuracy_score(y_val, y_pred)
                accuracy_sum += accuracy
                
            print(this_feat)    
            accuracy_average = accuracy_sum /5
            print(accuracy_average)
            if accuracy_average < worst_perf:
                worst_feat = feat
                worst_perf = accuracy_average
            if accuracy_average > best_score:
                best_feat_sel = this_feat.copy()
                best_score = accuracy_average
        feat_sel.remove(worst_feat)
        print(best_feat_sel)
        print(len(feat_sel))
        print(best_score)
    return best_feat_sel, best_score

def forwardSequentialFeatureSelect3Sec(X, y, k, samples_per_song, max_features=None):
    '''
    Selects the best feature set for k-NN based on accuracy using forward sequential feature selection.

    Parameters:
    - X: Features
    - y: Labels
    - k: k for k-NN
    - samples_per_song: Dictionary song index -> (number of samples, index of start of song in dataframe)
    - max_features: Maximum number of features to select (default is None, meaning all features)

    Returns:
    - best_feat_sel: Best selected feature set
    - best_score: Accuracy score of the best feature set
    '''

    if max_features is None or max_features > X.shape[1]:
        max_features = X.shape[1]

    feat_sel = []
    best_feat_sel = []
    best_score = 0

    while len(feat_sel) < max_features:
        best_feat = None
        best_perf = 0

        for feat in set(range(X.shape[1])) - set(feat_sel):
            this_feat = feat_sel + [feat]
            accuracy_sum = 0

            for X_train, X_val, y_train, y_val in utils.n_fold_cv.n_fold_cv(X, y, samples_per_song, n=5):
                X_train_subset = X_train[:, this_feat]
                X_val_subset = X_val[:, this_feat]
                
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_subset, y_train)
                
                y_pred = knn.predict(X_val_subset)
                accuracy = accuracy_score(y_val, y_pred)
                accuracy_sum += accuracy

            accuracy_average = accuracy_sum / 5
            print("Selected Features:", this_feat)
            print(accuracy_average)
            if accuracy_average > best_perf:
                best_feat = feat
                best_perf = accuracy_average

        feat_sel.append(best_feat)

        if best_perf > best_score:
            best_feat_sel = feat_sel.copy()
            best_score = best_perf
        print("Selected Features:", feat_sel)
        print("Number of Features:", len(feat_sel))
        print("Best Accuracy:", best_perf)
        print("---")


    return best_feat_sel, best_score
