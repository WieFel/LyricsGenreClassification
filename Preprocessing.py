import pandas, os, time
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, WordNetLemmatizer
from contraction_helper import expand_contractions_in_text
import cld2

DATA_PATH = os.path.expanduser("~/NLP_Data/")
ORIGINAL_DATA_SET = DATA_PATH + "final_55plus.csv"
FILTERED_DATA_SET = DATA_PATH + "filtered_lyrics.csv"
FINAL_OUTPUT = DATA_PATH + "dataset_55plus.npy"

# categories we want to extract
CATEGORY_DICT = {"country": ["country"], "religious": ["christian", "praise", "worship", "gospel"],
                 "hip-hop": ["hip-hop", "hip hop", "rap"], "metal": ["metal"], "jazz": ["jazz"]}
ALL_CATEGORIES = [g for l in CATEGORY_DICT.values() for g in l]


# detects the language of the passed text
# returns the language as a string (e.g. "ENGLISH") or None, if no language is detectable
def detect_language(text):
    detection = cld2.detect(text)
    if not detection.is_reliable or not detection.details:
        # detection of language not reliable or detected languages empty -> return None
        return None
    else:
        return detection.details[0].language_name


def resume_genre(genre):
    g = ""
    for c in ALL_CATEGORIES:
        if c in genre:
            g = c
            break

    for key in CATEGORY_DICT:
        if g in CATEGORY_DICT[key]:
            return key
    return ""


# reads the CSV data and does some basic filtering
def read_genre_lyrics_data():
    # read csv data, assuming the lyrics.csv is in ~/NLP_Data/
    csv_data = pandas.read_csv(ORIGINAL_DATA_SET)

    d = csv_data[["genre", "text"]]  # only use columns "genre" and "text" (lyrics)
    d = d[d["text"].notnull()]  # filter out rows with empty lyrics
    d = d[d["genre"].notnull()]  # filter out rows with undetected genres
    d["genre"] = d["genre"].map(str.lower)  # convert all genres to lowercase

    # # detect the languages of the lyrics and add it as new column
    d["language"] = d["text"].map(detect_language)

    d = d[(d.language == "ENGLISH")]  # only take english songs
    # extract only genres (tags) we want
    d = d[(d.genre.str.contains("country")) |
          (d.genre.str.contains("christian")) |
          (d.genre.str.contains("praise")) |
          (d.genre.str.contains("worship")) |
          (d.genre.str.contains("gospel")) |
          (d.genre.str.contains("hip-hop")) |
          (d.genre.str.contains("hip hop")) |
          (d.genre.str.contains("rap")) |
          (d.genre.str.contains("metal")) |
          (d.genre.str.contains("jazz"))
          ]

    d["genre"] = d["genre"].map(resume_genre)

    d = d[["genre", "text"]]  # again select only the two relevant columns genre and text
    d["text"] = d["text"].map(str.lower)  # convert lyrics to only lowercase
    return d


# expands the contractions within the lyric texts
def expand_contractions(data):
    data["text"] = data["text"].map(expand_contractions_in_text)


# tokenizes the lyrics texts
def tokenize(data):
    tokenizer = RegexpTokenizer(r'\w+')
    data["text"] = data["text"].map(tokenizer.tokenize)


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
    data["text"] = data["text"].map(pos_tag)  # pos-tagging of each word in each lyrics text


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
    data["text"] = data["text"].map(lambda word_postag_list: lemmatize_text(lemmatizer, word_postag_list))


# filters out the stopwords for every lyrics word array
def filter_stopwords(data):
    stop_words = set(stopwords.words('english'))  # stopwords as set to make lookups faster
    data["text"] = data["text"].map(lambda words: filter(lambda w: w not in stop_words, words))


# ------------------------- MAIN CODE ------------------------- #
if __name__ == "__main__":
    start = time.time()

    # FIRST STEP: read csv and filter out bad data
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
    np.save(FINAL_OUTPUT, data[["genre", "text"]].values)

    print("SUCCESS!")
    end = time.time()
    print("Time elapsed: " + str((end - start) / 60.0) + " min")
