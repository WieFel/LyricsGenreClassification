import pickle, os
import gensim


DATA_PATH = os.path.expanduser("~/NLP_Data/")
DATASET_FILE = DATA_PATH + "dataset.pickle"


# returns data as a np array with first column "genre" and second column "lyrics"
def read_dataset(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


# saves the word2vec model to a file in order to be used for classification
def save_model(model, filename):
    model.save(DATA_PATH + filename)


print("Reading dataset...")
data = read_dataset(DATASET_FILE)
genres = data["genre"].values
lyrics = data["lyrics"].values

print("Creating word2vec model...")
model = gensim.models.Word2Vec(lyrics, size=300, window=10, min_count=3, sample=1e-3)

# save model to file
save_model(model, "word2vec_model")