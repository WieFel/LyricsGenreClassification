import pandas, pickle, os, time
import pylast
import sys

DATA_PATH = os.path.expanduser("~/NLP_Data/")
ORIGINAL_DATA_SET = DATA_PATH + "lyrics1_english.csv"

API_KEY = "ca037ce35abbdf6f6ec8149d6fb293d2"
API_SECRET = "13c29f2c02d50b61f204dc6a670b42a6"
username = "xavi_diaz"
password_hash = pylast.md5("nlp_project3")



def get_genre(row):
    try:
        response = network.get_track(row.Band, row.Song).get_top_tags()
        print(row.name)
    except:
        row["genre"] = None
        return row
    
    if response: 
        row["genre"] = response[0].item
    else:
        row["genre"] = None
    return row
    
    
network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET,
                               username=username, password_hash=password_hash)
csv_data = pandas.read_csv(ORIGINAL_DATA_SET)

d = csv_data[["Band","Song","Lyrics"]]
d = d[int(sys.argv[1]):int(sys.argv[2])]
d = d.apply(get_genre, axis=1)
d.to_csv("lyricsEnglishWgenre" + sys.argv[1]+ ":" + sys.argv[2] + ".csv")