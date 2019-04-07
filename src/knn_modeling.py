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

# # Audio Processing
# import librosa
# import librosa.display
# from IPython.display import Audio
# from scipy.io import wavfile
# from pydub import AudioSegment
# from src.audio_processing import load_mp3_from_url


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
        
        
        
class ProductionValue():
    """
    The ProductionValue class. General class responsible for creating data frames from MongoDB, fitting models, plotting, and querying. A catch-all class.
    """
    
    def __init__(self, collection, sp):
        self.collection = collection
        self.sp = sp
        
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
        
    def query(self, track, artist=None, album=None):
        """
        Takes song info (track, artist, album) and returns 5 nearest neighbors for songs and 5 most likely producers.
        """
        query_df = self.song_df[
                          (self.song_df['track']==track) &
                          (self.song_df['artist']==artist if artist else 1) &
                          (self.song_df['album']==album if album else 1)
                         ]
        audio_data = query_df.iloc[0][self.audio_feature]
        flat_audio_data = np.array(audio_data)[:,:1200].flatten().reshape(1,-1)
        X_query_pca = self.pipeline.transform(flat_audio_data)
        top_producers = self.y_columns[np.argsort(np.vstack(self.knn.predict_proba(X_query_pca))[:,1])[::-1]]
        # NEED TO ADD PROBABILITIES

        distances, indices = self.knn.kneighbors(X_query_pca)
        top_songs = self.song_df.loc[indices.flatten().tolist()[:5]][['track','artist','album','producer']]
        top_songs['distance'] = distances.flatten()[:5]
        return top_producers, top_songs

        #NEED TO ADD FUNCTIONALITY IF SONG IS NOT IN DF
