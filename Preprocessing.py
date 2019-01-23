import pandas, pickle
import langid, contractions
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import time

# number of samples to work with in total
NUMBER_SAMPLES = 100000


def read_genre_lyrics_data():
    # read csv data, assuming the lyrics.csv is in ~/NLP_Data/
    csv_data = pandas.read_csv("/home/felix/NLP_Data/lyrics.csv")

    d = csv_data[["genre", "lyrics"]]  # only use columns "genre" and "lyrics"
    d = d[d["lyrics"].notnull()]   # filter out empty lyrics
    d["lyrics"] = d["lyrics"].apply(lambda t: unicode(t, errors="ignore"))   # convert lyrics characters to unicode

    # # detect the languages of the lyrics (first 100 chars of lyrics) and add it as new column
    d["language"] = d["lyrics"].map(lambda t: langid.classify(t[:100])[0])  # -> may take several minutes!

    d = d[(d.language == "en")]  # filter out english songs only
    d = d[["genre", "lyrics"]]  # again select only the two relevant columns genre and lyrics
    d = d.iloc[:NUMBER_SAMPLES]
    return d.values


# expands the contracted words for every lyrics text in the column (vector) of lyric texts
def expand_contractions_d1(column):
    for i, v in enumerate(column):
        column[i] = contractions.fix(v)


# expands the contractions within the texts present in the column 1 of the passed data array and returns it
def expand_contractions(data):
    np.apply_along_axis(expand_contractions_d1, 0, data[:, 1])     # axis 0 column 1 contains the lyrics texts


# tokenizes the lyrics texts present in the passed column vector and returns it
def tokenize_column_d1(column):
    tokenizer = RegexpTokenizer(r'\w+')
    for i, lyrics in enumerate(column):
        column[i] = tokenizer.tokenize(lyrics)


# tokenizes the lyrics column in the data array and returns the updated array
def tokenize(data):
    np.apply_along_axis(tokenize_column_d1, 0, data[:, 1])    # axis 0 column 1 contains the lyrics


# filters out the stop words from the word arrays present in the column and returns the new column
def filter_stopwords_column_d1(column):
    stop_words = set(stopwords.words('english'))    # stopwords as set to make lookups faster
    for i, words in enumerate(column):
        column[i] = filter(lambda w: w not in stop_words, words)  # filter out stopwords from list


# filters out the stopwords for every lyrics word array and returns the updated data array
def filter_stopwords(data):
    np.apply_along_axis(filter_stopwords_column_d1, 0, data[:, 1])    # axis 0 column 1 contains the lyrics


# writes the passed data to the file "filename"
def write_to_file(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


# FIRST STEP: read csv and filter out bad data. only work with a smaller dataset size of NUMBER_SAMPLES
start = time.time()
print("Reading data from original csv and filtering it...")
data = read_genre_lyrics_data()
end = time.time()
print("Time elapsed: " + str(end-start))

# SECOND STEP: Expand contractions within the lyrics texts
print("Expanding contractions in each lyrics text...")
expand_contractions(data)
print("Time elapsed: " + str(time.time()-end))
end = time.time()

# THIRD STEP: Tokenize lyrics texts to get lists of single words
# here, a lyrics text is discomposed to a single list of words, ignoring the sentence structure
# -> maybe, sentence structure could be relevant?
print("Tokenizing lyrics text into lists of words")
tokenize(data)
print("Time elapsed: " + str(time.time()-end))
end = time.time()

# FOURTH STEP: Filter out stopwords from word lists
print("Filtering stopwords from word lists")
filter_stopwords(data)
print("Time elapsed: " + str(time.time()-end))
end = time.time()

print("Writing data to file...")
write_to_file("/home/felix/NLP_Data/dataset.pickle", data)

print("SUCCESS!")
end = time.time()
print("Total time elapsed: " + str(end-start))
print(data)
