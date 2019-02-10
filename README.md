# LyricsGenreClassification
In this project, we try to train a neural network to discover the genre of a music track only by looking 
at its lyrics text.

## Data set
The data set we used can be found <a href="https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics">on Kaggle</a>.<br/>
It contains a total of 362.237 songs with additional information.<br/>
The data contains the following columns:<br/>
<ul>
    <li>index</li>
    <li>song</li>
    <li>year</li>
    <li>artist</li>
    <li>drake</li>
    <li>genre</li>
    <li>lyrics</li>
</ul>

We use the two columns <strong>genre</strong> and <strong>lyrics</strong>. The data set contains songs in various 
languages and some of the data-samples don't contain any lyrics. Thus, some preprocessing is necessary.

## Preprocessing
The preprocessing consists of several steps in order to convert the plaintext of the song-lyrics into a format that 
is processable by a neural network.

#### 1. Read data and filter for relevant data
The .csv-file contains 362.237 songs. Filtering out the songs with empty lyrics, 266.557 songs remain. Also, we only 
want to work with english songs. Therefore, we use the Python package <i>langid</i> to generate a new data column 
named "language". Using this new column, we discover the language of each song and remove all non-english songs. 
remove non-english tracks. After that, our data set has 240.643 songs with the following distribution over the genres:

| Genre             | Number samples |
| ----------------- | :------------: |
| Hip-Hop           | 22.422         |
| Metal             | 22.134         |
| Country           | 14.273         |
| Jazz              | 7.428          |
| Electronic        | 7.261          |
| R&B               | 3.321          |
| Indie             | 2.996          |
| Folk              | 1.849          |
| ~~Rock~~          | 101.997        |
| ~~Pop~~           | 34.489         |
| ~~Not Available~~ | 18.497         |
| ~~Other~~         | 3.976          |

The genres "Not Available" and "Other" are removed. Also, we remove the genres "Rock" and "Pop" because we think that they
might not be very distinguishable when looking only at the lyrics. Thus, we remain with a total of 81.684 songs. All the remaining lyrics are transformed to lowercase strings.

On a machine with 8GB of RAM, a quad core Intel Core i7 CPU @ 3.40 GHz and running Manjaro Linux, the execution 
of this first step took 1:30 minutes.

#### 2. Expand contractions
In this step, all the lyrics texts are taken and the contractions are expanded. So the following text
```
in better days i've been known to listen
i go to waste all my time is missing
i'm mapping out my ending ...
```
changes to
```
in better days i have been known to listen
i go to waste all my time is missing
i am mapping out my ending ...
```

This step, on the same machine took around 2:25 minutes for the whole 81.684 songs.

#### 3. Tokenization
The lyrics, which are still encoded as normal strings, are now tokenized ignoring the sentence- or 
newline-structure and simply considering the single words in their original order. The text from the previous
example would then look like this:

```
[in, better, days, i, have, been, known, to, listen, i, go,to, waste,
all, my, time, is, missing, i, am, mapping, out, my, ending, ...]
```

The tokenization only took about 6 seconds for the whole data set.

#### 4. POS tagging and lemmatization
In this step, the tokenized lyrics texts are taken, first POS tagged and then lemmatized. The POS tagging 
converts the example text from above to pairs of `(word, POS tag)`:
```
[(in, IN), (better, JJR), (days, NNS), (i, VBP), (have, VBP), (been, VBN), 
(known, VBN), (to, TO), (listen, VB), (i, JJ), (go, VBP), (to, TO), 
(waste, VB), (all, DT), (my, PRP$), (time, NN), (is, VBZ), (missing, VBG),
(i, NN), (am, VBP), (mapping, VBG), (out, RP), (my, PRP$), (ending, VBG), ...]
```
and then after lemmatization back to single words:
```
[in, good, day, i, have, be, know, to, listen, i, go, to, waste,
all, my, time, be, miss, i, be, map, out, my, end, ...]
```
POS tagging and lemmatization are the most time consuming tasks in the preprocessing pipeline and 
together took about 19:30 minutes for all the samples.

#### 5. Filter stopwords
The last step is to filter out the stopwords from the lyrics texts as they don't contain any valuable 
information we need. Removing the stopwords makes the texts quite shorter than before, as can be seen
in the example text:
```
[good, day, know, listen, go, waste, time, miss, map, end, ...]
```
Removal of stopwords took around 45 seconds.

In the end, the preprocessed data set is written to a pickle file, so that it can later be easily used with a 
neural network by simply loading it from the file, without having to preprocess the data again. The data set is 
written to the pickle file as pandas dataframe structure and thus can also be loaded as such. It has the 
following structure:
```
genre    lyrics
Hip-Hop  [horn, chorus, timbo, hit, phone, betta, know, ...
Hip-Hop  [hey, yo, man, remember, shit, eazy, back, mot ...
Hip-Hop  [yo, dre, man, take, bitch, movie, shit, man, ...
Hip-Hop  [aah, one, song, kick, back, smoke, joint, get ...
Hip-Hop  [hey, yo, man, remember, shit, eazy, back, mot ...
Hip-Hop  [artist, master, p, album, ghetto, dope, song, ...
Country  [get, mind, heart, belong, oh, could, make, al ...
Country  [well, look, like, doin, givin, love, money, g ...
Country  [never, think, would, hurt, bad, seein, look, ...
Country  [leave, tell, would, let, haunt, mind, would, ...
```
The first column contains the genre of the song and the second column contains a list of words 
representing the lyrics.
