# LyricsGenreClassification
In this project, we try to train a neural network to discover the genre of a music track only by looking 
at its lyrics text.

## Data set
Regarding the datasets used in this project, we found in two papers a link pointing to a compilation of 
"55000+ song-lyrics in English from LyricsFreak", made by user Kuznetsov. Unfortunately, the songs in the dataset were 
not labeled with its corresponding musical genres, so we had to fetch ourselves the labels from an external site, 
namely the last.fm website, which provided us with the tags we were looking for. 
Ultimately, we decided to discard too general genres like pop, rock etc., and focus only on genres that are a priori 
much easier to distiguish, i.e. metal, hip-hop, country, religious and jazz. After this filtering process, 
the original 55.000 dataset shrinked to less than 10.000. Because of this and because we thought a bigger dataset 
not used in the above mentioned papers before would be ideal, we then searched for another corpus in kaggle and 
found one named "Every song you have heard (almost). "Over 500,000 song lyric urls for over a million artists" of which 
just the first part/half of 250.000 was considered, as it was very time-consuming to preprocess all of the data. 
Once again the songs were not labelled, so this had to be done as well thanks to the api service of last.fm and after 
filtering the data to obtain just the songs with the genres of interest for our use case, this dataset shrinked to 
about 30.000 that in the end also showed better results in the project implementation.<br/>

Links to the two datasets:
<a href="https://www.kaggle.com/mousehead/songlyrics">55000+ Song Lyrics</a>
<a href="https://www.kaggle.com/artimous/every-song-you-have-heard-almost">Every song you have heard (almost)!</a>


## Preprocessing
The preprocessing consists of several steps in order to convert the plaintext of the song-lyrics into a format that 
is processable by a neural network.

#### 1. Read data and filter for relevant data
The .csv-file of the dataset contains around 250.000 songs. We only 
want to work with english songs. Therefore, we detect the languages of the lyircs using the python package `cld2` and 
remove all non-english songs from the dataset. After filtering out genres which are not interesting for us, 
the dataset has a size of 30.138 songs. The genres we are interested in are listed below with their respective 
quantities of songs.

| Genre             | Number samples |
| ----------------- | :------------: |
| Metal             | 9.147          |
| Country           | 7.852          |
| Jazz              | 5.860          |
| Hip-Hop           | 5.420          |
| Religious         | 1.859          |

#### 2. Expand contractions
In this step, all the lyrics texts are taken and the contractions are expanded.

#### 3. Tokenization
The lyrics, which are still encoded as normal strings, are now tokenized ignoring the sentence- or 

#### 4. POS tagging and lemmatization
In this step, the tokenized lyrics texts are taken, first POS tagged and then lemmatized. The POS tagging 
converts the example text from above to pairs of `(word, POS tag)`. After the POS tagging, lemmatization is applied and 
the text is transferred back to a normal list of words representation.

#### 5. Filter stopwords
The last step is to filter out the stopwords from the lyrics texts as they don't contain any valuable 
information we need. Removing the stopwords makes the texts quite shorter than before

In the end, the preprocessed data set is written to a .npy-file, so that it can be easily used for later steps 
by simply loading it from the file, without having to preprocess the data again. The data set is 
written to the file as a numpy array, where the first column contains the genre and the second column contains the 
list of words representing the lyrics of the song.

## Feature Extraction
In the feature extraction, the dataset from the preprocessing is read and numerical features are extracted.<br/>
A TF-IDF vectorizer object is created from the lyrics data as well as a word2vec model. Together, the two models are 
used to generate a numerical vector representation of each lyric-text, using 75 dimensions.

Those vectors are again written to a .npy-file and can then be used by a neural network.

## Model training
We tried two different approaches for training our neural network for lyrics-genre classification:
- use an LSTM network directly on the preprocessed text-data (implemented in `ClassificationLSTM.py`)
- apply feature extraction to the preprocessed text-data and feed the numerical vectors into a fully connected 
neural network (implemented in `Classification.py`)

After several tests and modifications, we came to the conclusion that the second mentioned approach yields a better 
accuracy and is also way faster to train.

The optimal parameters for the fully connected neural network were:
- Learning rate: 0.01
- Batch size: 256
- Number of steps: 20.000
- 2 layers of 32 and 64 neurons

The model achieved an average accuracy of 74% using 5-Fold cross-validation.

## Prediction
In the end, one should be able to make actual predictions with the created model. Therefore we stored the final model 
in a file in order to allow to load it at a later point in time. I.e. the weights and biases are stored to later make 
a prediction within the `Prediction.py` file. In this file, a lyrics text can be defined. This text is then 
preprocessed, converted to numerical vectors and can be fed into the neural network.<br/>
Due to time-reasons, we could not finish the very last step of loading the stored model and doing an actual prediction 
for the defined lyrics-string.