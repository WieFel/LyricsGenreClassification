import pickle
from FeatureExtraction import VECTORIZED_DATA


# returns data as pandas dataframe
def read_data(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


# load dataset
print("Reading vectorized data...")
data = read_data(VECTORIZED_DATA)

print(data)