import pandas
import pickle
import langid


def read_genre_lyrics_data():
    # read csv data, assuming the lyrics.csv is in ~/NLP_Data/
    csv_data = pandas.read_csv("/home/felix/NLP_Data/lyrics.csv")

    d = csv_data[["genre", "lyrics"]]  # only use columns "genre" and "lyrics"
    d = d[d["lyrics"].notnull()]   # filter out empty lyrics
    d["lyrics"] = d["lyrics"].apply(lambda t: unicode(t, errors="ignore"))   # convert lyrics characters to unicode

    # detect the languages of the lyrics (first 100 chars of lyrics) and add it as new column
    d["language"] = d["lyrics"].map(lambda t: langid.classify(t[:100])[0])  # -> may take several minutes!

    d = d[(d.language == "en")]  # filter out english songs only
    d = d[["genre", "lyrics"]]  # again select only the two relevant columns genre and lyrics
    return d.values



data = read_genre_lyrics_data()

with open("/home/felix/NLP_Data/dataset.pickle", "wb") as file:
    pickle.dump(data, file)

print("SUCCESS!")
