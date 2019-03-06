import pandas, pickle, os, time
import pylast

DATA_PATH = os.path.expanduser("~/NLP_Data/")
ORIGINAL_DATA_SET = DATA_PATH + "songdata.csv"

API_KEY = "4e9b3c33f00564ad07368e862acc27a5"
API_SECRET = "d7c05f73a54b2295208e6e703da442f9"
username = "robxcm"
password_hash = pylast.md5("gatito-1234")



def get_genre(row):
    try:
        response = network.get_track(row.artist, row.song).get_top_tags()
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

d = csv_data[["artist","song","text"]]
d = d.apply(get_genre, axis=1)
d.to_csv("datasetWgenre.csv")