import pandas

csv_data = pandas.read_csv("~/NLP_Data/lyrics.csv")   # read csv data
data = csv_data[["genre", "lyrics"]]  # only use columns "genre" and "lyrics"
data = data[data["lyrics"].notnull()]   # filter out empty lyrics
print(data.head)

# [362237 rows x 2 columns] TOTAL
# [266557 rows x 2 columns] WITH EMPTY LYRICS FILTERED OUT