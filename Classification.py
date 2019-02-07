import pickle, os


DATA_PATH = os.path.expanduser("~/NLP_Data/")
DATASET_FILE = DATA_PATH + "dataset.pickle"


# returns data as a np array with first column "genre" and second column "lyrics"
def read_dataset(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


data = read_dataset(DATASET_FILE)
print(data)