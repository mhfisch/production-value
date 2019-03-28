import numpy as np
import pandas as pd
from collections import defaultdict
import h5py
import os
import sys
import h5py
from importlib import reload


def h5_to_dict(f):
    """
    input:
        f: opened .h5 file from MSDS
    output:
        d: a dictionary version of the .h5 file
    """

# f is structured like a double-indexed dictionary.
    d = defaultdict(dict)

    # Create first layer of the multi-index
    for key in list(f.keys()):
        d[key] = defaultdict(dict)

        # Create the second layer of the multi-index and assign values
        for key2 in list(f[key]):
            d[key][key2] = list(f[key][key2])

            # Convert binary text to ascii
            for i, item in enumerate(d[key][key2]):
                if type(item) == np.bytes_:
                    d[key][key2][i] = d[key][key2][i].decode()

# Trying to convert d['metadata']['songs'] items from binary, but it doesn't work
    #             if type(item) == np.void:
    #                 for j, entry in enumerate(item):
    #                     if type(entry) == np.bytes_:
    #                         d[key][key2][i][j] = d[key][key2][i][j].decode()

    return d

def multi_indexer(d):
    """
    input:
        d: a MSDS dictionary from h5_to_dict()
    output:
        multi_index_dict: a multi-indexed dictionary compatible with Pandas DataFrames. Each double-indexed key is converted to a tuple.
    """
    multi_index_dict = {}
    for i, key in enumerate (d.keys()):
        for j, key2 in enumerate(d[key].keys()):
            multi_index_dict[(key,key2)] = [d[key][key2]]
    return multi_index_dict


def list_h5(walk_dir):
    """
    Makes a list of filepath strings for every .h5 file in directory

    INPUT:
    walk_dir: string filepath to a directory

    OUTPUT:
    file_list: a list of .h5 filepath strings
    """

    file_list = []
    for root, subdirs, files in os.walk(walk_dir):

        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path[-2:] == 'h5':
                file_list.append(file_path)

    return file_list


def dir_to_h5df(walk_dir, N):
    """
    Converts the first N .h5 files in a directory (or its subdirectories) to a Pandas DataFrame

    INPUTS:
    walk_dir: STR path to the root directory of the files
    N: number of files to put in the directory. If N = 'all', all files will be converted.

    OUTPUTS:
    h5df: PANDAS DATAFRAME where each row is the information in an .h5 file
    """

    h5_file_list = list_h5(walk_dir)

    if (N == 'all') or (N > len(h5_file_list)):
        files_to_convert = h5_file_list
    else:
        files_to_convert = h5_file_list[:N]

    # Convert list of files names to list of dictionaries

    h5_df_list = []

    for filename in files_to_convert:
        f = h5py.File(filename, 'r')
        h5_df = pd.DataFrame(multi_indexer(h5_to_dict(f)))
        h5_df_list.append(h5_df)

    h5df = pd.concat(h5_df_list, ignore_index=True)

    return h5df



def find_producer(track, album, artist, year, discogs_token, N=10):
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



def add_producer(d, track, album, artist, year, discogs_token, N=10):
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
