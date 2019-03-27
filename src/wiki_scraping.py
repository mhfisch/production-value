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
    
    song_links = soup.find_all('div', class_="mw-category")[0].find_all('a')
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



