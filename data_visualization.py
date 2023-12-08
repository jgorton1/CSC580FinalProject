'''
This is a messy file that was used for quick testing of different models on a simple validation set
I also used it to make visuals of the data.
PCA was shown unhelpful, and my features alone only resulted in 20% accuracy. Combining with the csv features generated from Librosa seems promising


'''
import numpy as np
from io import TextIOWrapper
from operator import add
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from typing import Any
import json as js

def main():
    training_file = open(".\\Data\\train_features.csv", "r")
    testing_file = open(".\\Data\\test_features.csv", "r")
    csv_file = open(".\\Data\\features_3_sec.csv")
    training_file.readline()
    testing_file.readline()
    classes: dict[str, tuple[list[float], list[float]]] = {}
    class_counter: dict[str, int] = {}
    train_data: list[list[float]] = []
    train_label: list[str | None] = []
    test_data: list[list[float]] = []
    test_label: list[str] = []
    csv_dict: dict[tuple[str, int, int], list[float]] = {}

    csv_file.readline()
    for x in csv_file.readlines():
        line = x.split(".")
        x = x.split(",")
        key: tuple[str, int, int] = line[0], int(line[1]), int(line[2])
        csv_dict[key] = [float(y) for y in x[1: -1]]
        
    counter = 0
    missed: list[int] = []
    for x in training_file.readlines():
        features: list[float] = []
        melody, harmony, label, number, partition = get_data_from_line(x)
        melody = [float(val) for val in melody]
        harmony = [float(val) for val in harmony]

        #if(label in classes):
        #    temp_m, temp_h = classes[label] 
        #    classes[label] = (list(map(add, temp_m, melody)), list(map(add, temp_h, harmony)))
        #    class_counter[label] += 1
        #else:
        #    classes[label] = (melody, harmony)
        #    class_counter[label] = 1
        
        if((label, number, partition) in csv_dict):
            features = csv_dict[(label, number, partition)][:50] + melody #+ harmony comparable to having just harmony (more separable data)
            train_label.append(label)
            train_data.append(features)
        else:
            missed.append(counter)
        counter += 1


    for x in testing_file.readlines():
        features: list[float] = []
        melody, harmony, label, number, partition = get_data_from_line(x)
        melody = [float(val) for val in melody]
        harmony = [float(val) for val in harmony]

        features = harmony

        if((label, number, partition) in csv_dict):
            test_label.append(label)
            test_data.append(csv_dict[(label, number, partition)][:50] + melody)

    for x in classes.keys():
        temp_m, temp_h = classes[x]
        size = class_counter[x]
        classes[x] = (list(map(lambda x: x / size, temp_m)), list(map(lambda x: x / size, temp_h)))

    
    
    x_train, y_train, x_valid, y_valid = get_validation(missed, train_data, train_label)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid= scaler.fit_transform(x_valid)

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    outputs = []
    for x in range(1, 5):
        knn = KNeighborsClassifier(n_neighbors=x)
        knn.fit(train_data, train_label)
        output = knn.predict(test_data)

        correct = accuracy_score(output, test_label)
        print(x, correct)
    
    #numbering = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal":6, "pop": 7, "reggae": 8, "rock":9}
    #new_train_labels = [numbering[x] for x in train_label]
    #attempt_nn(train_data, new_train_labels)
    

    #data, labels = cg()
    #attempt_nn(data, labels)
    
    

        

    '''
    figure, axis = plt.subplots(2, 5)
    counter: int = 0
    for key in classes.keys():
        axis[counter // 5, counter % 5].set_title(f"Average {key} melody")
        axis[counter // 5, counter % 5].set_ylim(0, .25)
        axis[counter // 5, counter % 5].xlabel = "Melodic Difference"
        axis[counter // 5, counter % 5].ylabel = "Relative Frequency"
        axis[counter // 5, counter % 5].bar(range(0, 7), classes[key][0])
        
        counter += 1
    plt.show()
    figure, axis = plt.subplots(2, 5)
    counter: int = 0
    for key in classes.keys():
        print(key)
        axis[counter // 5, counter % 5].bar(range(0, 35), classes[key][1])
        axis[counter // 5, counter % 5].set_title(f"Average {key} harmony")
        axis[counter // 5, counter % 5].set_ylim(0, .18)
        axis[counter // 5, counter % 5].xlabel = "Harmonic Combination"
        axis[counter // 5, counter % 5].ylabel = "Relative Frequency"
        counter += 1
    plt.show()
    '''
    
def attempt_nn(training_data: list[list[float]], training_labels: list[int]):
    x1, y1, x2, y2 = get_validation(training_data, training_labels)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(10,), random_state=1, max_iter=10000)
    scaler = StandardScaler().fit(x1)
    x1 = scaler.transform(x1)
    scaler = StandardScaler().fit(x2)
    x2= scaler.transform(x2)

    clf.fit(np.array(x1), y1)
    output = clf.predict(x2)
    correct = accuracy_score(output, y2)
    print(correct)


def cg():
    numbering = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal":6, "pop": 7, "reggae": 8, "rock":9}
    cgfile = open(".\\Data\\chronogram.json")
    cgdata = js.load(cgfile)
    labels: list[int] = []
    out_data = []
    for x in cgdata:
        for y in cgdata[x]:
            for z in cgdata[x][y]:
                # X is train/test, y is genre, z is number
                data = np.array(cgdata[x][y][z])
                for i in range(0, 10):
                    out_data.append(cg_get_features(data[:, i*15:(i+1)*15]))

                    labels.append(numbering[y])
    
    return np.array(out_data), np.array(labels)

def cg_get_features(data: list[list[float]]):
    temp: list[float] = []
    for x in data:
        for y in x:
            temp.append(y)
    return np.array(temp)



def get_validation(counter, training_data: list[list[float]], labels: list[Any]):
    train: list[list[float]] = []
    train_labels: list[str] = []
    test: list[list[float]] = []
    test_labels: list[str] = []
    subtracter = 0
    for x in range(0, 10):
        for y in range(0, 720):
            if((x+1) * y in counter):
                subtracter += 1
                continue
            train.append(training_data[(x + 1) * y - subtracter])
            train_labels.append(labels[(x + 1) * y - subtracter])
        for y in range(720, 800):
            test.append(training_data[(x + 1) * y - subtracter])
            test_labels.append(labels[(x + 1) * y - subtracter])
    
    return train, train_labels, test, test_labels
        



def attempt_pca(data: list[list[float]]):
    scaler = StandardScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    #print(normalized_data)

    pca = PCA(n_components=3)
    pca.fit(normalized_data)
    final = pca.transform(normalized_data)
    colors = []
    for x in range(0, 10):
        for _ in range(0, 800):
            colors.append(x)
    print(pca.explained_variance_ratio_)

    axis = plt.figure(figsize=(10, 10)).add_subplot(111, projection='3d')
    axis.scatter(final[:,0], final[:,1], final[:,2], c=colors, cmap="plasma")
    plt.show()

def get_data_from_line(text_line: str):
    line = text_line.strip().split(",")
    melodic_vals = line[-8:-1]
    harmonic_vals = line[1:-8]
    label = line[-1]
    filename = line[0].split(".")
    number = int(filename[1])
    partition = int(filename[2])



    return melodic_vals, harmonic_vals, label, number, partition

main()
