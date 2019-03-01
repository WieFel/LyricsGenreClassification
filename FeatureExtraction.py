import pickle, os, time
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from Preprocessing import DATA_PATH, FINAL_OUTPUT

VECTORIZED_DATA = DATA_PATH + "vectorized_data.pickle"
DIMENSIONS = 300


# returns data as pandas dataframe
def read_dataset(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


# identity function for tokenizer
def identity_tokenizer(text):
    return text


# calculates the tfidf vectors of the given corpus and returns the vectorizer and the feature vectors
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range,
                                 # the following parameters are necessary for processing list of words as input
                                 analyzer='word',
                                 tokenizer=identity_tokenizer,
                                 preprocessor=identity_tokenizer,
                                 token_pattern=None)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# function to compute tfidf weighted averaged word vector for a document
# this code has been taken from the book "Text Analysis with Python"
def tfidf_wtd_avg_word_vector(words, tfidf_vector, tfidf_vocabulary, model, num_features):
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)] if tfidf_vocabulary.get(word) else 0 for word in words]
    word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    feature_vector = np.zeros((num_features,), dtype="float64")
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)
    return feature_vector


# generalize above function for a corpus of documents
def tfidf_weighted_averaged_word_vectorizer(documents, tfidf_vectors, tfidf_vocabulary, model, num_features):
    docs_tfidfs = [(doc, doc_tfidf) for doc, doc_tfidf in zip(documents, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vector(tokenized_sentence, tfidf, tfidf_vocabulary, model, num_features)
                for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)


# writes the passed data to the file "filename"
def write_to_file(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


# saves the word2vec model to a file in order to be used for classification
def save_model(model, filename):
    model.save(DATA_PATH + filename)


# ------------------------- MAIN CODE ------------------------- #
if __name__ == "__main__":
    t1 = time.time()  # only for time measure purpose

    print("Reading dataset...")
    data = read_dataset(FINAL_OUTPUT)
    genres = data["genre"].values
    lyrics = data["lyrics"].values

    print("Creating TF-IDF model...")
    tfidf_vectorizer, tfidf_features = tfidf_extractor(lyrics)

    print("Creating word2vec model...")
    model = gensim.models.Word2Vec(lyrics, size=DIMENSIONS, window=10, min_count=3, sample=1e-3)
    save_model(model, "word2vec_model")  # save model to file (for eventual visualization)

    # transform information to feature matrix
    print("Creating feature data...")
    feature_matrix = tfidf_weighted_averaged_word_vectorizer(data["lyrics"].values, tfidf_features,
                                                             tfidf_vectorizer.vocabulary_, model, DIMENSIONS)

    # store all the data inside one matrix with columns "feature-1 | feature-2 | ... | feature-n | genre-label"
    vectorized_data = np.hstack(
        (feature_matrix, genres.reshape((len(genres), 1))))  # horizontally stacks feature matrix and labels

    print("Writing vector data to file...")
    write_to_file(VECTORIZED_DATA, vectorized_data)

    print("Success!")
    print("Time elapsed: " + str((time.time() - t1) / 60.0) + " min")
