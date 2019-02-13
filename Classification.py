import os
import numpy as np
from gensim.models import Word2Vec


DATA_PATH = os.path.expanduser("~/NLP_Data/")
WORD2VEC_MODEL_FILE = DATA_PATH + "word2vec_model"


# load our gensim
model = Word2Vec.load(WORD2VEC_MODEL_FILE)

# convert the wv word vectors into a numpy matrix that is suitable for insertion
# into our TensorFlow and Keras models
embedding_matrix = np.zeros((len(model.wv.vocab), 300)) # last parameter = dimension
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


print(embedding_matrix)