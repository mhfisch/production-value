# Standard Imports
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict
from importlib import reload
from bs4 import BeautifulSoup
import requests
import scipy.stats as scs
import time
import seaborn as sns
from sklearn.pipeline import Pipeline
import pickle

import matplotlib.pyplot as plt

# Modeling
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import datasets, layers, models
# import keras.backend as K
# from tensorflow.keras.constraints import min_max_norm, non_neg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Load MongoDB
from pymongo import MongoClient
client = MongoClient()
# Access/Initiate Database
db = client['producer_db']
# Access/Initiate Table
mfcc_tab = db['mfcc']
tab = db['songs']
collection = db.tab
mfcc_collection = db.mfcc_tab

# Authorize Spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
client_id = os.environ['SPOTIFY_CLIENT_ID']
client_secret = os.environ['SPOTIFY_CLIENT_SECRET']
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Discogs token
discogs_token = os.environ['DISCOGS_TOKEN']

# # Audio Processing
import librosa
import librosa.display
from IPython.display import Audio
import IPython.display
from scipy.io import wavfile
from pydub import AudioSegment
from src.audio_processing import load_mp3_from_url, mfcc_highpass

# Plotly
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import colorlover as cl
from IPython.display import HTML
plotly_username = os.environ['PLOTLY_USERNAME']
plotly_api_key = os.environ['PLOTLY_API_KEY']
plotly.tools.set_credentials_file(username=plotly_username, api_key=plotly_api_key)
#set colormap
Set3_10 = cl.scales['10']['qual']['Set3']
cm_10 = list(zip(np.linspace(0,1,10),Set3_10)) 


def plot_mnist_embedding(ax, X, y, title=None, alpha = 1):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.
      
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(y[i]), 
                 color=plt.cm.tab10((y[i]) / 10.), 
                 fontdict={'weight': 'bold', 'size': 16},
                 alpha = alpha)

    ax.set_xticks([]), 
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize = (9,9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def evaluate_model(collection, audio_feature = 'MFCC'):
    """
    Prints accuracy vs benchmark metric, a confusion matrix, and a t-SNE plot
    INPUTS: collection (MongoDB Collection)
            audio_feature (STR), the name of the audio feature of interest in the collection.
    OUTPUTS: None
    """
    
    # Create a Pandas DataFrame of the Data
    song_data_list = []
    for song in collection.find():
        try:
            song_data = (song['track'],song['artist'],song[audio_feature],song['basic_genre'],song['producer'])
            song_data_list.append(song_data)
        except:
            pass
    song_df = pd.DataFrame(song_data_list, columns = ['track','artist',audio_feature,'genre','producer'])
    song_df = song_df[song_df[audio_feature].apply(lambda x: np.array(x).shape[1]) >= 1200] #ensures all audio features are same size.

    # Make feature matrix
    X = np.vstack(song_df[audio_feature].apply(lambda x: np.array(x)[:,:1200].flatten()).values)
    
    # Make target vector, one-hot encoded vector, and y_columns - a "legend" for the vectors.
    y = song_df['producer']
    y_one_hot = pd.get_dummies(y).values
    y_columns = np.array(pd.get_dummies(y).columns)

    # Test train split.
    X_train, X_test, y_train, y_test = train_test_split(X ,y_one_hot, test_size = 0.3, random_state = 440)

    # Standardize
    ss = StandardScaler()
    X_train_scale = ss.fit_transform(X_train)
    X_test_scale = ss.transform(X_test)


    # Use PCA to reduce dimensionality
    pca = PCA(n_components=12)
    X_train_pca = pca.fit_transform(X_train_scale)
    X_test_pca = pca.transform(X_test_scale)

    # Make a knn model
    knn = KNeighborsClassifier(n_neighbors = 30)
    knn.fit(X_train_pca, y_train)

    # Convert y labels from unit vectors to integers
    y_test_labels = np.argmax(y_test, axis = 1)
    y_hat = knn.predict(X_test_pca)
    y_hat_labels = np.argmax(np.stack(knn.predict_proba(X_test_pca))[:,:,1].T, axis = 1)


    # Calculate Metrics
    accuracy = (y_test_labels == y_hat_labels).sum()/y_test_labels.size
    maj_class = np.argmax(y_train.sum(axis = 0))
    benchmark = (y_test_labels == maj_class).sum()/y_test_labels.size # Benchmark is guessing majority class
    
    # Print Metric
    print("Producer Accuracy:\t{:.2f}".format(accuracy))
    print("Benchmark:\t\t{:.2f}".format(benchmark)) 
    print()


    # Plot non-normalized confusion matrix
    np.set_printoptions(precision=2)
    plt.figure(figsize = (12,12))
    plot_confusion_matrix(y_test_labels, y_hat_labels, classes=y_columns,
                          title='Confusion matrix: KNN Model using {}'.format(audio_feature))
    plt.show()

    # Do t-SNE Transform
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_train_scale)
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, learning_rate=20, n_iter = 5000)
    X_tsne = tsne.fit_transform(X_train_pca, np.argmax(y_train, axis=1))

    # Plot t-SNE
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_mnist_embedding(ax, X_tsne, np.argmax(y_train, axis = 1), alpha = 0.7)
    plt.show()
    
    # Print Legend
    print()
    print('t-SNE Plot Legend')
    print()
    for i, producer in enumerate(y_columns):
        print(i, producer)

        
"""
-----------------------------------------------
FUCTIONS FOR FINDING PRODUCERS FROM DISCOGS API
-----------------------------------------------
"""        
        
def find_producer(discogs_token, track, album='', artist='', year='', N=10):
    """
    Takes four strings: artist, track, album, and discogs token and returns the tuple (role='Producer', name, and discogs resource_url),
    if one is returned in the first N results on discogs.

    INPUTS:
        track: STR - the name of the song
        album: STR - the name of the album
        artist: STR - the name of the artist
        year: STR - the year the song was released
        discogs_token: STR - API string for api.discogs.com
        N: INT - number of results to iterate through before giving up on finding a producer
    """

    # Use Discogs API to search for artist, track, album
    artist = artist.replace(' ','+')
    track = track.replace(' ','+')
    album = album.replace(' ','+')
    api_query = requests.get('https://api.discogs.com/database/search?track={}&artist={}&release_title={}&year={}&type=release&token={}'
                             .format(track, artist, album, year, discogs_token)).json()['results']

    # api_query is a LIST. Do the following:

    producer_list = []
    # search api_query for 'role' = 'Producer'
    for i, entry in enumerate(api_query):
        for producer in gen_key_value_extract('role', entry, 'producer', ['role','name','resource_url']):
            producer_list.append(producer)

        # check if we've found a producer yet
        if len(producer_list) > 0:
            return producer_list

        # if no Producer, GET api_subquery from the 'resource_url' field
        resource_url = list(gen_dict_extract('resource_url', entry))[0]
        api_subquery = requests.get(resource_url).json()

        # search next_api_query for 'role' = 'Producer'
        for producer in gen_key_value_extract('role', api_subquery, 'producer', ['role','name','resource_url']):
            producer_list.append(producer)

        # check if we've found a producer yet
        if len(producer_list) > 0:
            return producer_list

        # If no producer, loop back and go to the next item in api_query.
        if i>=N:
            return producer_list


def gen_key_value_extract(key, var, value, req_keys):
    """
    In a nested dictionary, var, where value in key, return req_value for keys in req_keys.

    INPUT:
        key: OBJECT - The key to match in a nested dictionary
        var: DICT - The nested dictionary to iterate through
        value: OBJECT - The desired matching value to `key`
        req_keys: LISTLIKE - a list of requested keys whose values will be returned

    OUTPUT:
        result: GENERATOR OBJECT - result returns tuples of values associated with the keys in req_keys
    """
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                if value.lower() in v.lower():
                    v_list = []
                    for req_key in req_keys:
                        v_list.append(var[req_key])
                    v_tup = tuple(v_list)
                    yield v_tup
            if isinstance(v, dict):
                for result in gen_key_value_extract(key, v, value, req_keys):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_key_value_extract(key, d, value, req_keys):
                        yield result


def gen_dict_extract(key, var):
    """
    Creates a generator object that returns all of the matching values for a `key` in a nested dictionary, `var`

    INPUT:
        key: OBJECT - key to match in nested dictionary
        var: DICT - nested dictionary

    OUTPUT:
        result: GENERATOR - generates the values wherever a key = `key` in the nested dictionary.

    """
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result



def add_producer(d, discogs_token, track, album='', artist='', year='', N=10):
    """
    Adds a 'producer' key to a dictionary, d, inplace whose corresponding value is a set of tuples containing (producer_id, producer_name) pairs from the discogs API. The producer_id comes from the discogs database url.

    INPUTS:
        d: DICT - Dictionary to append
        track: STR - the name of the song
        album: STR - the name of the album
        artist: STR - the name of the artist
        year: STR - the year the song was released
        discogs_token: STR - API string for api.discogs.com
        N: INT - number of results to iterate through before giving up on finding a producer

    OUPUTS:
        None
        Appends the current dictionary inplace

    """

    output = []
    producer_list = find_producer(track, album, artist, year, discogs_token, N=10)

    # Does role == 'Producer'
    for producer in producer_list:
        print(producer[1])
        if producer[0] == 'Producer':
            output.append(producer)
            print('Producer')

    # Does role contain 'Producer'
    if len(output) == 0:
        for producer in producer_list:
            print(producer[1])
            if 'Producer' in producer[0]:
                output.append(producer)
                print('...Producer...')

    # Does role contain 'produc'
    if len(output) == 0:
        for producer in producer_list:
            print(producer[1])
            if 'produc' in producer[0]:
                output.append(producer)
                print('...produc...')

    # Add the set of likely producers to the input dictionary
    producer_set = set()
    for producer in output:
        producer_id = int(producer[2].replace('https://api.discogs.com/artists/',''))
        producer_set.add((producer_id, producer[1]))
    d['producers'] = producer_set

    
def find_one_producer(discogs_token, track, album='', artist='', year='', N=10):
    """
    Returns a string of the first producer for a song
    """
    producer_list = find_producer(discogs_token, track, album, artist, year, N)
    output = []

    # Does role == 'Producer'
    for producer in producer_list:
        if producer[0] == 'Producer':
            output.append(producer)

    # Does role contain 'Producer'
    if len(output) == 0:
        for producer in producer_list:
            if 'Producer' in producer[0]:
                output.append(producer)

    # Does role contain 'produc'
    if len(output) == 0:
        for producer in producer_list:
            if 'produc' in producer[0]:
                output.append(producer)
                
    return output[0][1]
    
"""
----------------------
PRODUCTION VALUE CLASS
----------------------
"""

class ProductionValue():
    """
    The ProductionValue class. General class responsible for creating data frames from MongoDB, fitting models, plotting, and querying. A catch-all class.
    """
    
    def __init__(self, collection, sp, discogs_token):
        # user input
        self.collection = collection
        self.sp = sp
        self.discogs_token = discogs_token
        
        # to be added later
        self.audio_feature = None
        self.song_df = None
        self.pipeline = None
        self.X_transform = None
        self.knn = None
        self.y_columns = None
        self.y_labels = None
        self.y_hat_labels = None
        self.accuracy = None
        self.plot_df = None
        
        
    def to_pickle(self, path):
        """
        Saves all local variables in pickle format in a dictionary
        """
        d = {}
        d['audio_feature'] = self.audio_feature
        d['song_df'] = self.song_df
        d['pipeline'] = self.pipeline
        d['X_transform'] = self.X_transform
        d['knn'] = self.knn
        d['y_columns'] = self.y_columns
        d['y_labels'] = self.y_labels
        d['y_hat_labels'] = self.y_hat_labels
        d['accuracy'] = self.accuracy
        d['plot_df'] = self.plot_df
        with open(path, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        
        pass
    
    def read_pickle(self, path):
        """
        Loads local variables in pickle format
        """
        
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.audio_feature = d['audio_feature']
        self.song_df = d['song_df']
        self.pipeline = d['pipeline']
        self.X_transform = d['X_transform']
        self.knn = d['knn'] 
        self.y_columns = d['y_columns']
        self.y_labels = d['y_labels']
        self.y_hat_labels = d['y_hat_labels']
        self.accuracy = d['accuracy']
        self.plot_df = d['plot_df']
        
    def fit_knn(self, audio_feature, n_neighbors=30, n_components=12):
        """
        Fit all the data in collection[audio_feature] to a KNN model after a PCA decomposition.
        INPUTS:
            audio_feature: STR name of MFCC audio feature from collection
            n_neighbors: INT number of nearest neighbors for KNN algorithm
            n_components: INT number of dimensions for PCA decomposition
        RETURNS:
            NONE
        """
        self.audio_feature = audio_feature
        song_data_list = []
        for song in self.collection.find():
            try:
                song_data = (song['track'],song['artist'],song['album'],song[audio_feature],song['basic_genre'],song['producer'])
                song_data_list.append(song_data)
            except:
                pass
        song_df = pd.DataFrame(song_data_list, columns = ['track','artist','album',audio_feature,'genre','producer'])
        song_df = song_df[song_df[audio_feature].apply(lambda x: np.array(x).shape[1]) >= 1200] #ensures all audio features are same size.
        self.song_df = song_df
        
        # Make feature matrix (n_samples, 20*1200)
        X = np.vstack(song_df[audio_feature].apply(lambda x: np.array(x)[:,:1200].flatten()).values)

        # Make target vector, one-hot encoded vector, and y_columns - a "legend" for the vectors.
        y = song_df['producer']
        y_one_hot = pd.get_dummies(y).values
        y_columns = np.array(pd.get_dummies(y).columns)
        self.y_columns = y_columns

        # Make Data Pipeline
        pipeline = Pipeline([('ss',StandardScaler()),
                     ('pca',PCA(n_components=n_components))
                    ]
                   )
        X_transform = pipeline.fit_transform(X)
        self.pipeline = pipeline
        self.X_transform = X_transform
        
        # Make a knn model
        knn = KNeighborsClassifier(n_neighbors = 30)
        knn.fit(X_transform, y_one_hot)
        self.knn = knn
        
        # Convert y labels from unit vectors to integers
        y_labels = np.argmax(y_one_hot, axis = 1)
        y_hat = knn.predict(X_transform)
        y_hat_labels = np.argmax(np.stack(knn.predict_proba(X_transform))[:,:,1].T, axis = 1)
        self.y_labels = y_labels
        self.y_hat_labels = y_hat_labels

        # Calculate Metrics
        accuracy = (y_labels == y_hat_labels).sum()/y_labels.size
        self.accuracy = accuracy
        
    def query(self, track, artist=None, album=None, use_spotify=False):
        """
        Takes song info (track, artist, album) and returns 5 nearest neighbors for songs and 5 most likely producers.
        """
        # NEED TO ADD .lower() functionality
        query_df = self.song_df[
                          (self.song_df['track']==track) &
                          (self.song_df['artist']==artist if artist else 1) &
                          (self.song_df['album']==album if album else 1)
                         ]
        # Check if song is in df
        if len(query_df.index) != 0:
            producer = query_df.iloc[0]['producer']
            audio_data = query_df.iloc[0][self.audio_feature]
            flat_audio_data = np.array(audio_data)[:,:1200].flatten().reshape(1,-1)
            X_query_pca = self.pipeline.transform(flat_audio_data)
            
            # Predict songs and producers
            producer_probabilities = np.vstack(self.knn.predict_proba(X_query_pca))[:,1]
            top_producers = self.y_columns[np.argsort(producer_probabilities)[::-1]]
            top_probabilities = producer_probabilities[np.argsort(producer_probabilities)[::-1]]
            producer_proba = np.stack([top_producers,top_probabilities]).T

            # Make top songs dataframe
            distances, indices = self.knn.kneighbors(X_query_pca)
            top_songs = self.song_df.loc[indices.flatten().tolist()[:5]][['track','artist','album','producer']]
            top_songs['distance'] = distances.flatten()[:5]
            return producer_proba, top_songs, producer
        
        # Check if song mp3 url is on spotify
        if use_spotify:
            producer_proba, top_songs, producer_discogs = self.query_spotify(track=track, 
                                                                            artist=artist, 
                                                                            album=album, 
                                                                            upsert=False
                                                                           )
            return producer_proba, top_songs, producer_discogs
        
        else:
            print ('Track not found')
        
        
    def add(self, track, artist=None, album=None):
        """
        Adds a song to the mongoDB
        """
        
        top_producers, top_songs, producer_discogs = self.query_spotify(track=track, 
                                                                            artist=artist, 
                                                                            album=album, 
                                                                            upsert=True
                                                                           )
        return None
        
    def add_predict(self, track, artist=None, album=None):
        """
        Adds a song to the DB and then predicts the producer
        """     
        top_producers, top_songs, producer_discogs = self.query_spotify(track=track, 
                                                                            artist=artist, 
                                                                            album=album, 
                                                                            upsert=True
                                                                           )
        return top_producers, top_songs, producer_discogs
        
    def predict(self, M):
        """
        Given an MFCC matrix (20,1200+), predicts producer and nearest songs
        """
        flat_audio_data = np.array(M)[:,:1200].flatten().reshape(1,-1)
        X_query_pca = self.pipeline.transform(flat_audio_data)
        
        # Predict songs and producers
        producer_probabilities = np.vstack(self.knn.predict_proba(X_query_pca))[:,1]
        top_producers = self.y_columns[np.argsort(producer_probabilities)[::-1]]
        top_probabilities = producer_probabilities[np.argsort(producer_probabilities)[::-1]]
        producer_proba = np.stack([top_producers,top_probabilities]).T

        distances, indices = self.knn.kneighbors(X_query_pca)
        top_songs = self.song_df.loc[indices.flatten().tolist()[:5]][['track','artist','album','producer']]
        top_songs['distance'] = distances.flatten()[:5]
        return producer_proba, top_songs

    def query_spotify(self, track, artist=None, album=None, upsert=False):
        """
        Take a query search (track, artist, album) and returns
        (top_producers(LIST), top_songs(PANDAS DATAFRAME), and producer_discogs(STR)
        If upsert=True, the song is added to the database.
        """
        query = track + (' artist:{}'.format(artist) if artist else '') + (' album:{}'.format(album) if album else '')
        print(query)
        search = sp.search(q=query, type='track')
        song = search['tracks']['items'][0]

        # song info
        preview_url = song['preview_url']
        # check for mp3
        if not preview_url:
            print('No audio file available for track:'+query)
            return None, None, None
        self.query_preview_url = preview_url

        # more song info
        song_id = song['id']
        track = song['name']
        album = song['album']['name']
        artist = song['artists'][0]['name']
        artist_id = song['artists'][0]['id']

        # Lookup producer on Discogs
        try:
            producer_discogs = find_one_producer(self.discogs_token, track, album=album, artist=artist, year='', N=10)
        except:
            producer_discogs = None
            upsert = False

        # Get genre list from artist
        genre_list = sp.artist(artist_id)['genres']

        # Use preview_url to get audio processing
        y, sr = load_mp3_from_url(preview_url)
        M = mfcc_highpass(y, sr)

        # Pipeline audio data
        flat_audio_data = np.array(M)[:,:1200].flatten().reshape(1,-1)
        X_query_pca = self.pipeline.transform(flat_audio_data)

        # Predict songs and producers
        producer_probabilities = np.vstack(self.knn.predict_proba(X_query_pca))[:,1]
        top_producers = self.y_columns[np.argsort(producer_probabilities)[::-1]]
        top_probabilities = producer_probabilities[np.argsort(producer_probabilities)[::-1]]
        producer_proba = np.stack([top_producers,top_probabilities]).T

        # get top songs
        distances, indices = self.knn.kneighbors(X_query_pca)
        top_songs = self.song_df.loc[indices.flatten().tolist()[:5]][['track','artist','album','producer']]
        top_songs['distance'] = distances.flatten()[:5]
        pass

        # Add data to MongoDB
        if upsert:
            myquery = { "producer": producer_discogs,
                        "spotify_id" : song_id }

            newvalues = { "$set": {'mfcc_highpass':M.tolist(),
                                   'producer':producer_discogs,
                                   'spotify_id':song_id,
                                   'album':album,
                                   'artist':artist,
                                   'preview_url':preview_url,
                                   'track':track,
                                   'genres':genre_list}
                        }
            self.collection.update_one(myquery, newvalues, upsert = upsert)

        return producer_proba, top_songs, producer_discogs
    
    
    def plot_tsne(self):
        
        if type(self.plot_df) == type(None):
            # Make 2D t-SNE
            tsne2 = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, learning_rate=20, n_iter = 1000)
            X_tsne2 = tsne2.fit_transform(self.X_transform)

            # Create DF with plotting data
            self.plot_df = self.song_df[['track','artist','producer']]
            self.plot_df['tsne2_x'] = X_tsne2[:,0]
            self.plot_df['tsne2_y'] = X_tsne2[:,1]
            self.plot_df['labels'] = '<b>Producer: ' + self.plot_df['producer'] + '</b><br>Track: '+ self.plot_df['track'] + '<br>Artist: ' + self.plot_df['artist']
        
        data = []

        for i, producer in enumerate(self.y_columns):

            trace = go.Scatter(
                x = self.plot_df[self.plot_df['producer']==producer]['tsne2_x'],
                y = self.plot_df[self.plot_df['producer']==producer]['tsne2_y'],
                name = producer,
                mode = 'markers',
                marker = dict(size = 10,
                              color = cm_10[i][1],
                              line = dict(width = 1)),
                text = self.plot_df[self.plot_df['producer']==producer]['labels'],
                hoverinfo = 'text',
            )

            data.append(trace)


        layout = dict(title = 't-SNE Plot of Tracks',
                      yaxis = dict(zeroline = False,
                                   showline = False,
                                   ticks = '',
                                   showticklabels = False,
                                   showgrid = False),
                      xaxis = dict(zeroline = False,
                                   showline = False,
                                   ticks = '',
                                   showticklabels = False,
                                   showgrid = False),
                      hovermode = 'closest'
                     )

        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='2-D t-SNE Plot')
        
    def demo(self, track, artist=None, album = None):
        producer_proba, top_songs, producer = self.query(track, artist, album, use_spotify=True)
        if type(top_songs) == type(None):
            print ('Song mp3 not available. :-( ')
            return producer_proba, top_songs, producer
        
        # get top song info
        query_preview_url = self.query_preview_url
        top_track = top_songs['track'].iloc[0]
        top_artist = top_songs['artist'].iloc[0]
        top_song_url = self.collection.find_one({'track':top_track})['preview_url']
        
        # process audio:
        y_query, sr_query = load_mp3_from_url(query_preview_url)
        y_top, sr_top = load_mp3_from_url(top_song_url)
        
        if type(producer) == type(None):
            print("True Producer: Not Found")
        else:
            print("True Producer: {}".format(producer))
        print ()
        print ("Producer Probabilities:")
        print (producer_proba)
        print ()
        print ("Query Song: {} by {}".format(track, artist))
        IPython.display.display(Audio(data = y_query, rate = sr_query))
        print ()
        print ("Most Similar Song: {} by {}".format(top_track, top_artist))
        IPython.display.display(Audio(data = y_top, rate = sr_top))
        
        return producer_proba, top_songs, producer
            