import os, pickle
import numpy as np
from gensim.models import Word2Vec


DATA_PATH = os.path.expanduser("~/NLP_Data/")
DATASET_FILE = DATA_PATH + "dataset.pickle"
WORD2VEC_MODEL_FILE = DATA_PATH + "word2vec_model"

DIMENSIONS = 300


# returns data as pandas dataframe
def read_dataset(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def sum_up_word_vecs(lyrics):
    v = np.zeros((DIMENSIONS,))
    for word in lyrics:
        try:
            v += model[word]
        except:
            pass
    return v    # return sum vector


# load dataset
print("Reading dataset...")
data = read_dataset(DATASET_FILE)

# load our gensim
model = Word2Vec.load(WORD2VEC_MODEL_FILE)

print(data)

data["lyrics"] = data["lyrics"].map(sum_up_word_vecs)

print("NEW DATA:")
print(data)