import pickle
import numpy as np


# returns data as a np array with first column "genre" and second column "lyrics"
def read_dataset(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)




data = read_dataset("/home/felix/NLP_Data/dataset.pickle")
print(data)