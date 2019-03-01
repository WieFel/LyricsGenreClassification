import pandas, pickle, os, time
import langid
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, WordNetLemmatizer
from contraction_helper import expand_contractions_in_text


DATA_PATH = os.path.expanduser("~/NLP_Data/")
ORIGINAL_DATA_SET = DATA_PATH + "lyrics.csv"
FILTERED_DATA_SET = DATA_PATH + "filtered_lyrics.csv"
FINAL_OUTPUT = DATA_PATH + "dataset.pickle"


# reads the CSV data and does some basic filtering
def read_genre_lyrics_data():
    # read csv data, assuming the lyrics.csv is in ~/NLP_Data/
    csv_data = pandas.read_csv(ORIGINAL_DATA_SET)

    d = csv_data[["genre", "lyrics"]]  # only use columns "genre" and "lyrics"
    d = d[d["lyrics"].notnull()]  # filter out rows with empty lyrics
    d["lyrics"] = d["lyrics"].map(lambda t: unicode(t, errors="ignore"))  # convert to unicode because of langid

    # # detect the languages of the lyrics (first 100 chars of lyrics) and add it as new column
    d["language"] = d["lyrics"].map(lambda t: langid.classify(t[:100])[0])  # -> may take several minutes!

    d = d[(d.language == "en")]  # filter out english songs only
    d = d[["genre", "lyrics"]]  # again select only the two relevant columns genre and lyrics
    d = d[(d.genre != "Not Available") & (d.genre != "Other")]  # filter out genres "Not Available" and "Other"
    d = d[(d.genre != "Rock") & (d.genre != "Pop")]  # ignore genres Rock and Pop

    d["lyrics"] = d["lyrics"].map(str)  # convert back to normal string to apply lowercase
    d["lyrics"] = d["lyrics"].map(str.lower)  # convert lyrics to only lowercase
    return d


# expands the contractions within the lyric texts
def expand_contractions(data):
    data["lyrics"] = data["lyrics"].map(lambda lyrics: expand_contractions_in_text(lyrics))


# tokenizes the lyrics texts
def tokenize(data):
    tokenizer = RegexpTokenizer(r'\w+')
    data["lyrics"] = data["lyrics"].map(tokenizer.tokenize)


# converts between the POS-tag syntax used by POS-tagging and the syntax used by the lemmatizer
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# applies the pos_tag function of nltk to every lyrics word-list
# after applying this function, the lyrics-column contains lists of (word, pos-tag) pairs
def pos_tagging(data):
    data["lyrics"] = data["lyrics"].map(pos_tag)  # pos-tagging of each word in each lyrics text


# lemmatizes one lyrics text represented as list of (word, pos-tag) pairs
def lemmatize_text(lemmatizer, word_postag_list):
    return map(
        lambda word_postag: lemmatizer.lemmatize(word_postag[0], get_wordnet_pos(word_postag[1])) if get_wordnet_pos(
            word_postag[1]) else word_postag[0], word_postag_list)


# lemmatizes every lyrics word-list
# this function collapses the (word, pos-tag) pairs generated in the pos_tagging-function
# to the simple lemmas of the words
def lemmatize(data):
    lemmatizer = WordNetLemmatizer()
    data["lyrics"] = data["lyrics"].map(lambda word_postag_list: lemmatize_text(lemmatizer, word_postag_list))


# filters out the stopwords for every lyrics word array
def filter_stopwords(data):
    stop_words = set(stopwords.words('english'))  # stopwords as set to make lookups faster
    data["lyrics"] = data["lyrics"].map(lambda words: filter(lambda w: w not in stop_words, words))


# writes the passed data to the file "filename"
def write_to_file(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


# ------------------------- MAIN CODE ------------------------- #
if __name__ == "__main__":
    start = time.time()

    # FIRST STEP: read csv and filter out bad data. only work with a smaller dataset size of NUMBER_SAMPLES
    if not os.path.isfile(FILTERED_DATA_SET):
        # filtered data file doesn't exist -> generate it
        print("Filtering original data set...")
        data = read_genre_lyrics_data()
        print("Writing filtered data to CSV...")
        data.to_csv(FILTERED_DATA_SET)
    else:
        # filtered data file exists -> read it
        print("Reading filtered data set...")
        data = pandas.read_csv(FILTERED_DATA_SET)

    # SECOND STEP: Expand contractions within the lyrics texts
    print("Expanding contractions...")
    expand_contractions(data)

    # THIRD STEP: Tokenize lyrics texts to get lists of single words
    print("Tokenizing...")
    tokenize(data)

    # FOURTH STEP: POS-tag and lemmatize
    print("POS-tagging...")
    pos_tagging(data)

    print("Lemmatizing...")
    lemmatize(data)

    # FIFTH STEP: Filter out stopwords from word lists
    print("Removing stopwords...")
    filter_stopwords(data)

    print("Writing data to file...")
    write_to_file(FINAL_OUTPUT, data)

    print("SUCCESS!")
    end = time.time()
    print("Time elapsed: " + str((end - start)/60.0) + " min")
