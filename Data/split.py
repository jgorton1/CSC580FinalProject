import os
import shutil
import random

# Define the path to the directory containing genre folders
dataset_dir = "genres_original/"

# Define the path for the train and test directories
train_dir = "train"
test_dir = "test"

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all the genre folders in the dataset directory
genres = os.listdir(dataset_dir)

# Loop through each genre folder
for genre in genres:
    genre_path = os.path.join(dataset_dir, genre)
    
    # List all files in the genre folder
    files = os.listdir(genre_path)
    
    # Shuffle the list of files
    random.shuffle(files)
    
    # Calculate the split point based on the 80/20 ratio
    split_point = int(0.8 * len(files))
    
    # Split the files into train and test sets
    train_files = files[:split_point]
    test_files = files[split_point:]
    
    # Create genre subdirectories in the train and test directories
    train_genre_dir = os.path.join(train_dir, genre)
    test_genre_dir = os.path.join(test_dir, genre)
    os.makedirs(train_genre_dir, exist_ok=True)
    os.makedirs(test_genre_dir, exist_ok=True)
    
    # Copy the files to the train and test genre directories
    for file in train_files:
        src = os.path.join(genre_path, file)
        dst = os.path.join(train_genre_dir, file)
        shutil.copy(src, dst)
    
    for file in test_files:
        src = os.path.join(genre_path, file)
        dst = os.path.join(test_genre_dir, file)
        shutil.copy(src, dst)

print("Dataset split completed with an 80/20 ratio.")
