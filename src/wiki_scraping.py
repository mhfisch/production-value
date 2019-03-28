"""
A set of functions useful for scraping track metadata from Wikipedia.
"""

import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict
from importlib import reload
from bs4 import BeautifulSoup
import requests


def load_producers(cat_url_list, collection, sp):
    
    for cat_url in cat_url_list:

        #Extract Producer Name
        html = requests.get(cat_url).content
        soup = BeautifulSoup(html, 'html.parser')
        producer = soup.find_all('h1', {'id':"firstHeading"})[0].text.split('by ')[-1]

        print('-'*20)
        print('PRODUCER: {}'.format(producer))
        print('-'*20)
        print()

        #Scrape Wikipedia Page for Songs and get spotify track id's
        print('Scraping Wikipedia')
        spotify_info = get_spotify_info_from_wiki(cat_url, sp)

        print('Example data:')
        for i in range(5):
            print(spotify_info[i])
        print()

        print('Extracting Audio Analysis...')
        print()

        idx_list = []

        #Use SpotiPy to access song featurized data
        for track, artist, album, song_id, spotify_track, spotify_artist in spotify_info:
            print('Importing {} by {}...'.format(track,artist))
            query = 'track:{} artist:{}'.format(track,artist)
            result = sp.search(q=query, type='track')
            song_id = result['tracks']['items'][0]['id']
            song_info = sp.track(song_id)
            song_analysis = sp.audio_analysis(song_id)
            song_features = sp.audio_features(song_id)


            #Add featurized data to MongoDB
            new_entry = {'track':track,
                         'artist':artist,
                         'album':album,
                         'producer':producer,
                         'spotify_id':song_id,
                         'track_info':song_info,
                         'audio_analysis':song_analysis,
                         'audio_features':song_features}

            idx = collection.insert_one(new_entry)
            idx_list.append(idx)

            print('Import Complete.')
            print()
            
    return idx_list

def get_spotify_info_from_wiki(cat_url, sp):
    """
    Returns a LIST of TUPLES in the form (track, artist, album, spotify_track_id, spotify_track, spotify_artist)
    
    INPUT:
        cat_url: STR - path to wikipedia category page
        sp: SPOTIPY OBJECT with verified credentials
        
    OUTPUT:
        spotify_info: LIST OF TUPLES OF STRING in form (track, artist, album, spotify_track_id, spotify_track, spotify_artist)
    """
    print("Scraping Wikipedia")
    print()
    song_info_list = get_wiki_from_category(cat_url)
    
    spotify_info = []

    print("Querying Spotify API")
    print()
    for track, artist, album in song_info_list:
        try:
            query = 'track:{} artist:{}'.format(track,artist)
            results = sp.search(q=query, type='track')
            song_id = results['tracks']['items'][0]['id']
            spotify_track = results['tracks']['items'][0]['name']
            spotify_artist = results['tracks']['items'][0]['artists'][0]['name']
            new_song_info = (track, artist, album, song_id, spotify_track, spotify_artist)
            spotify_info.append(new_song_info)
        except:
            print('{} by {} not found.'.format(track, artist))
            pass

    return spotify_info



def get_wiki_from_category(category_url):
    """
    Returns a list of (track, artist, album) tuples for every song listed in a Wikipedia Category page such as 
    'https://en.wikipedia.org/wiki/Category:Song_recordings_produced_by_George_Martin'
    
    INPUTS:
        category_url: STR - path to wikipedia category page
        
    OUTPUTS:
        song_info_list: LIST of TUPLES of STRINGS - LIST of (track, artist, album) TUPLES for every song linked in a category page.
    
    """
    
    song_urls = []
    domain = 'https://en.wikipedia.org'
    
    html_cat = requests.get(category_url).content
    soup_cat = BeautifulSoup(html_cat, 'html.parser')
    
    song_links = soup_cat.find_all('div', class_="mw-category")[0].find_all('a')
    for link in song_links:
        path = link['href']
        url = domain + path
        song_urls.append(url)
           
        #NEED TO ADD FUNCTIONALITY FOR MULTIPLE PAGES
        
    song_info_list = []
    
    for song_url in song_urls:
        try:
            song_info = get_wiki_song_info(song_url)
            song_info_list.append(song_info)
        except:
            pass
        
    return song_info_list



def get_wiki_song_info(song_url):
    """
    Extract Track title, Artist Name, and Album Name from a Wikipedia entry for a song.
    
    INPUT:
    song_url: STR - url for wikipedia entry for a song
    
    OUTPUT:
    song_info: TUPLE of STR - (track, artist, album)
    """
    
    html_song = requests.get(song_url).content

    soup_song = BeautifulSoup(html_song, 'html.parser')

    track = soup_song.find_all(
        'table', class_='infobox vevent')[0].find_all(
        'th', class_='summary')[0].text.replace('"','')

    artist = soup_song.find_all(
        'table', class_='infobox vevent')[0].find_all(
        'th', class_='description')[0].text.split('by ')[-1]

    album = soup_song.find_all(
        'table', class_='infobox vevent')[0].find_all(
        'th', class_='description')[1].text.replace('from the album ','')

    return track, artist, album



