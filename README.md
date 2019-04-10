<img src="images/production_value_logo.png" align="center" />

# Production Value
> Identifying Record Producers from Audio Data

## Table of Contents

* [Motivation](#motivation)
  * [Personal](#personal)
  * [Business](#business)
* [Data Understanding](#data-understanding)
  * [Data Sources](#data-sources)
  * [Audio Processing](#audio-processing)
  * [Modeling](#modeling)
  * [Evaluation](#evaluation)
* [Future Improvements](#future-improvements)
* [Built With](#built-with)
* [Acknowledgements](#acknowledgements)


## Motivation

### Personal

<img src="images/bill_hare.png" align="center" width = "500" />

In my senior year of college I lead my a cappella group in the creation of our album [Just in Capes](link). I networked with Grammy Award-winning producer [Bill Hare](link) and saw firsthand the extent to which a music producer can have influence over the sound of an album.

### Business

This model can be used for two main purposes:
* **Music Discovery**: Services like Pandora and Spotify leverage their ability to find music users will like. Production "sound" is another dimension that users may enjoy exploring when searching for new music.
* **Music Publishing**: There currently exists no database of music credits. When streaming services pay royalties to record labels often creative collaborators do not get paid properly because of missing documentation. Production Value is a step toward "fingerprinting" the creators of a song.

## Data Understanding

### Data Sources

* [Spotify API](link) - Contains audio files and song metadata.
* [Wikipedia](wikipedia.org) - Record producer labeling.

### Audio Processing

Identifying a record producer lies in the [timbre](wikilink) of a sound. Timbre can be thought of as the "quality" or "identity" of a sound. It's what allows us to tell a flute from a trumpet even if they are playing the same notes. Timbre can be found in the higher-frequency [overtones](wikilink) of a sound.

Audio mp3 clips 30-seconds long from 1000 songs (10 producers, 100 songs each) were converted to .WAV files and run through a highpass filter to accentuate the timbre frequencies. For each clip, the [Mel-Frequency Cepstral Coefficients](wikilink) (MFCCs) were calculated.

MFCCs, very generally, are a set of values that correspond to the timbre of a sound.

More technically, MFCCs are calculated by first taking the [Fast Fourier Transform](wikilink) (FFT) of a waveform to convert from amplitude-time space to frequency-time space. Then, each frequency power spectrum of the FFT is treated as its own wavelet and is decomposed further using the [Discrete Cosine Transform](wikilink) (DCT). The resulting values are the Mel-Frequency Cepstral Coefficients. The figure below shows an example of the audio processing.

<img src="images/audio_processing.png" align="center" />

### Modeling

After processing, each song has about 24,000 MFCCs (20 in the frequency dimension, 1200 in the time dimension). [Principal Component Analysis](wikilink) (PCA) was used to reduce the dimensionality to 12 sonic eigenvectors.

A [K-Nearest Neighbors](wikilink) (KNN) algorithm was used to identify the most likely producers for any new song. The figure below shows how an example of how the KNN algorithm works.

<img src="images/data_pipeline.png" align="center" />

### Evaluation

The model was tested on a 300-song testing set. The multiclass accuracy for 10 balanced classes of producers was 44% compared to a baseline of 10%.

<img src="images/confusion_matrix.png" align="center" />

The data were then plotted on a 2D [t-SNE](link) plot to show the relative clustering of songs.

<img src="images/tsne.png" align="center" />

An interactive t-SNE plot can be found [here](plotly).

## Future Improvements
* Deconvolution of Variables:
  * Artist/Album/Instrumentation
  * More accurate labeling
* Scale:
  * More songs/producers
  * Parallelize and deploy on AWS/Spark
* Feature Engineering:
  * More Audio Processing/Reverse Engineering
  * Remove music structure by breaking songs into beats
* Modeling:
  * Neural Networks with Tensorflow/Keras


## Built With

* [Python](link)
* [MongoDB](link)
* [Pandas](link)
* [Numpy](link)
* [LibROSA](link)
* [SpotiPy](link)
* [SciPy](link)
* [SKLearn](link)
* [Plotly](link)

## Acknowledgements

* Everyone for everything!









<!-- # Important data

## What do the fields mean?

Find the field descriptions at [https://labrosa.ee.columbia.edu/millionsong/pages/field-list]


From www.discogs.com:

  In the music industry, a record producer has many roles, among them controlling the recording sessions, coaching and guiding the musicians, organizing and scheduling production budget and resources, and supervising the recording, mixing and mastering processes. This has been a major function of producers since the inception of sound recording, but in the latter half of the 20th century producers have also taken on a wider entrepreneurial role.


Maybe I should use NME.com's 50 of the Greatest Producers Ever [https://www.nme.com/list/50-of-the-greatest-producers-ever-1353]


Also check out [https://en.wikipedia.org/wiki/Record_producer#Influential_record_producers]

And [https://en.wikipedia.org/wiki/Category:Record_producers]

Audio Analysis Description [https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-analysis/]

Audio Features [https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/]

Note: Audio Valence in Audio Features is "A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."

Also, check out LibROSA [https://librosa.github.io/librosa/index.html] - Open source python audio analyzer.

### What is a Record Producer?

From www.discogs.com:

  >In the music industry, a record producer has many roles, among them controlling the recording sessions, coaching and guiding the musicians, organizing and scheduling production budget and resources, and supervising the recording, mixing and mastering processes. This has been a major function of producers since the inception of sound recording, but in the latter half of the 20th century producers have also taken on a wider entrepreneurial role.

### Why do this project?
[Insert my personal story here]

# Problem Statement

**Music Discovery:**
  >Reasons, reasons, reasons.

**Music Attribution in Publishing:**
  >More, more reasons.


 -->
