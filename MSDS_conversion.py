import numpy as np
import pandas as pd
from collections import defaultdict
import h5py


def h5_to_dict(f):
    """
    input:
        f: opened .h5 file from MSDS
    output:
        d: a dictionary version of the .h5 file
    """
    
#f is structured like a double-indexed dictionary.

    #Create first layer of the multi-index
    for key in list(f.keys()):
        d[key] = defaultdict(dict)
        
        #Create the second layer of the multi-index and assign values
        for key2 in list(f[key]):
            d[key][key2] = list(f[key][key2])
            
            #Convert binary text to ascii
            for i, item in enumerate(d[key][key2]):
                if type(item) == np.bytes_:
                    d[key][key2][i] = d[key][key2][i].decode()
                    
#Trying to convert d['metadata']['songs'] items from binary, but it doesn't work                    
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

