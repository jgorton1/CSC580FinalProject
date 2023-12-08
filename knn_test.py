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

        
        if((label, number, partition) in csv_dict):
            features = csv_dict[(label, number, partition)][:50] + melody
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

    
    
    # perhaps it is the fact that I use the standard scaler
    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    outputs = []
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(train_data, train_label)
    output = knn.predict(test_data)

    correct = accuracy_score(output, test_label)
    print(correct)

    
    

        

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
