"""
Python script with the function scrape_albums_from_wiki() that takes a list of Category:Albums produced by RECORD_PRODUCER and calls the Spotify API for featurized audio data and metadata for all of the songs in all of the albums listed in the Wiki Category page.

"""


# Standard Imports
import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict
from importlib import reload
from bs4 import BeautifulSoup
import requests
from time import sleep
import time
import matplotlib.pyplot as plt


# Load MongoDB
from pymongo import MongoClient
client = MongoClient()
# Access/Initiate Database
db = client['producer_db']
# Access/Initiate Table
tab = db['songs']
collection = db.tab

# Authorize Spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
client_id = os.environ['SPOTIFY_CLIENT_ID']
client_secret = os.environ['SPOTIFY_CLIENT_SECRET']
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_category_links(wiki_url):
    """
    Takes a link to a category Wikipedia page and returns a list of urls to the hyperlinks
    """
    
    wiki_urls = []
    domain = 'https://en.wikipedia.org'
    
    html = requests.get(wiki_url).content
    soup = BeautifulSoup(html, 'html.parser')
    
    wiki_links = soup.find_all('div', class_="mw-category")[0].find_all('a')
    for link in wiki_links:
        path = link['href']
        url = domain + path
        wiki_urls.append(url)
        
    # check for a "next page" button
    next_page_url = ''
    if soup.find_all('div', {'id':"mw-subcategories"}):
        next_page_links = soup.find_all('div', {'id':"mw-subcategories"})[0].find_all('a') #next page like will be within the first 5 links
        for link in next_page_links:
            if link.text == 'next page':
                next_page_path = link['href']
                next_page_url = domain + next_page_path
        
    # Append links from next pages recursively
    if next_page_url: 
        print('getting links from {}'.format(next_page_url))
        next_page_wiki_urls = get_category_links(next_page_url)
        wiki_urls = wiki_urls + next_page_wiki_urls
    
    
    return wiki_urls




def entry_from_wiki_album(album_url, sp, producer):
    
    """
    Function that takes album_url and a spotipy object and returns 
    OUTPUT: (track, artist, album, producer, spotify_id, track_info, audio_analysis, Audio_features)
    """
    
    album_html = requests.get(album_url).content
    alb_soup = BeautifulSoup(album_html, 'html.parser')
    
    album = alb_soup.find_all('th', class_='summary album')[0].text
    artist = alb_soup.find_all('div', class_='contributor')[0].text
    
    print()
    print('- '*10)
    print('Pinging Spotify for {} by {}...'.format(album,artist))
    print('- '*10)
    print()
    try:
        album_results = sp.search(q='album:{} artist:{}'.format(album, artist),type='album')
        
        # if there are no albums returned, try again without artist name
        if not album_results['albums']['items']:
            album_results = sp.search(q='album:{}'.format(album),type='album')
        
        album_id = album_results['albums']['items'][0]['id']

        album_output = []

        for song in sp.album_tracks(album_id)['items']:
            track = song['name']
            spotify_id = song['id']
            track_info = sp.track(spotify_id)
            audio_analysis = sp.audio_analysis(spotify_id)
            audio_features = sp.audio_features(spotify_id)

            album_output.append( (track, artist, album, producer, spotify_id, track_info, audio_analysis, audio_features) )
        return album_output
    

    except Exception as ex:
        print( ex )
        print(album_results)
        print()
        if album_results == None:
            sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            try:
                results = entry_from_wiki_album(album_url, sp, producer)
                return results
            except:
                raise Exception('Album {} not scraped'.format(album_url))
                
                
def scrape_album_list(producer_url):
    
    """
    INPUT: producer_url (wikipedia)
    OUTPUT: list of (album_url, producer) tuples
    """
    
    html = requests.get(producer_url).content
    soup = BeautifulSoup(html, 'html.parser')
    producer = soup.find_all('h1', {'id':"firstHeading"})[0].text.split('by ')[-1]
    print('Producer: {}'.format(producer))
    
    album_url_list = get_category_links(producer_url)
    
    output = list(zip( album_url_list, [producer]*len(album_url_list) ))
    
    return output

def scrape_albums_from_wiki(producer_urls, collection, sp):
    """
    Takes a list of wikipedia category:albums produced by PRODUCER pages and adds the featurized audio data and metadata from the Spotify API into a MongoDB. Returns a list of albums that could not be added.
    """
    
    failed_albums = defaultdict(list)
    for producer_url in producer_urls:
        album_list = scrape_album_list(producer_url)

        for i, (album_url, producer) in enumerate(album_list):
            try:
                album_entry_list = entry_from_wiki_album(album_url, sp, producer)
                print('\nalbum_entry_list length: {}'.format(len(album_entry_list)))
                if album_entry_list:
                    album_name = album_entry_list[0][2]
                print('-'*20)
                print('ALBUM: {} -- COMPLETE'.format(album_name))
                print('-'*20)
                print()
                for track, artist, album, producer, spotify_id, track_info, audio_analysis, audio_features in album_entry_list:
                    myquery = { "producer": producer,
                                "spotify_id" : spotify_id }

                    newvalues = { "$set": {'track':track,
                                           'artist':artist,
                                           'album':album,
                                           'producer':producer,
                                           'spotify_id':spotify_id,
                                           'track_info':track_info,
                                           'audio_analysis':audio_analysis,
                                           'audio_features':audio_features
                                          }
                                }

                    collection.update_one(myquery, newvalues, upsert = True)

                    print('\t\t{} by {}'.format(track,artist))
                print()

            except Exception as ex:
                failed_albums[producer].append((i,album_url,producer))
                print( ex )
                print()
                print('x'*20)
                print('ALBUM FAILURE: {}'.format(album_url))
                print('album_entry_list:')
                print(str(album_entry_list)[:1000])
                print()
                print('x'*20)
                print()
                
                
    return failed_albums

