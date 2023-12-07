from sklearn.model_selection import StratifiedKFold
def n_fold_cv(features, labels, samples_per_song, n=5):
    '''
    Does cross validation based on songs, rather than portions of songs
    Is a generator - if you want just one train test split, call next(n_fold_cv(...))
    '''
    #samples_per_song is a dictionary song index -> (number of samples, index of start of song in dataframe)
    #
    #num_samples = features.shape[0]
    #assert num_samples % 10 == 0
    #num_songs = num_samples // 10
    num_songs = len(samples_per_song)
    # Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    song_labels = [labels[samples_per_song[i][1]] for i in range(num_songs)]
    #print(song_labels)
    for train_index, test_index in skf.split(range(num_songs), song_labels):
        #print(test_index)
        train_indices = []
        for num in train_index:
            train_indices += [samples_per_song[num][1]+ i for i in range(samples_per_song[num][0]) ]
        test_indices = []
        for num in test_index:
            test_indices += [samples_per_song[num][1] + i for i in range(samples_per_song[num][0])]
        #print(test_indices)
        X_train, X_val = features[train_indices, :], features[test_indices, :]
        y_train, y_val = labels[train_indices], labels[test_indices]

        yield X_train, X_val, y_train, y_val