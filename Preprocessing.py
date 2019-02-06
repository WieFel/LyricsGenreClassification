import pandas, pickle, os
import langid, contractions
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import time


DATA_PATH = os.path.expanduser("~/NLP_Data/")
ORIGINAL_DATA_SET = DATA_PATH + "lyrics.csv"
FILTERED_DATA_SET = DATA_PATH + "filtered_lyrics.csv"


# reads the CSV data and does some basic filtering
def read_genre_lyrics_data():
    # read csv data, assuming the lyrics.csv is in ~/NLP_Data/
    csv_data = pandas.read_csv(ORIGINAL_DATA_SET)

    d = csv_data[["genre", "lyrics"]]  # only use columns "genre" and "lyrics"
    d = d[d["lyrics"].notnull()]   # filter out rows with empty lyrics
    d["lyrics"] = d["lyrics"].map(lambda t: unicode(t, errors="ignore"))   # convert lyrics characters to unicode

    # # detect the languages of the lyrics (first 100 chars of lyrics) and add it as new column
    d["language"] = d["lyrics"].map(lambda t: langid.classify(t[:100])[0])  # -> may take several minutes!

    d = d[(d.language == "en")]  # filter out english songs only
    d = d[["genre", "lyrics"]]  # again select only the two relevant columns genre and lyrics
    d = d[(d.genre != "Not Available") & (d.genre != "Other")]  # filter out genres "Not Available" and "Other"
    d = d[(d.genre != "Rock") & (d.genre != "Pop")]     # ignore genres Rock and Pop
    return d


# expands the contractions within the texts present in the column 1 of the passed data array and returns it
def expand_contractions(data):
    data["lyrics"] = data["lyrics"].map(lambda lyrics: contractions.fix(lyrics))


# tokenizes the lyrics column in the data array and returns the updated array
def tokenize(data):
    tokenizer = RegexpTokenizer(r'\w+')
    data["lyrics"] = data["lyrics"].map(tokenizer.tokenize)


# filters out the stopwords for every lyrics word array and returns the updated data array
def filter_stopwords(data):
    stop_words = set(stopwords.words('english'))  # stopwords as set to make lookups faster
    data["lyrics"] = data["lyrics"].map(lambda words: filter(lambda w: w not in stop_words, words))


# writes the passed data to the file "filename"
def write_to_file(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


# FIRST STEP: read csv and filter out bad data. only work with a smaller dataset size of NUMBER_SAMPLES
start = time.time()
if not os.path.isfile(FILTERED_DATA_SET):
    # filtered data file doesn't exist -> generate it
    print("Filtering original data set...")
    data = read_genre_lyrics_data()
    print("Writing filtered data to CSV...")
    data.to_csv(DATA_PATH + "filtered_lyrics.csv")
else:
    # filtered data file exists -> read it
    data = pandas.read_csv(FILTERED_DATA_SET)


# SECOND STEP: Expand contractions within the lyrics texts
# print("Expanding contractions in each lyrics text...")
# expand_contractions(data)
# print("Time elapsed: " + str(time.time()-end))
# end = time.time()
#
# # THIRD STEP: Tokenize lyrics texts to get lists of single words
# # here, a lyrics text is discomposed to a single list of words, ignoring the sentence structure
# # -> maybe, sentence structure could be relevant?
# print("Tokenizing lyrics text into lists of words")
# tokenize(data)
# print("Time elapsed: " + str(time.time()-end))
# end = time.time()
#
# # FOURTH STEP: Filter out stopwords from word lists
# print("Filtering stopwords from word lists")
# filter_stopwords(data)
# print("Time elapsed: " + str(time.time()-end))
# end = time.time()
#
# print("Writing data to file...")
# write_to_file(os.path.expanduser("~/NLP_Data/dataset.pickle"), data)
#
# print("SUCCESS!")
# end = time.time()
# print("Total time elapsed: " + str(end-start))
# print(data)
