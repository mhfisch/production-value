import urllib.request
from os import path
from pydub import AudioSegment
import librosa

def load_mp3_from_url(mp3_url, mp3_path = "temp.mp3", wav_path = "wav.mp3"):
    
    # retreive audio from url and save
    urllib.request.urlretrieve(mp3_url, mp3_path)

    # files                                                                         
    src = mp3_path
    dst = wav_path

    # convert mp3 to wav                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

    y, sr = librosa.load(wav_path)
    
    return y, sr


def get_mfcc(mp3_url):
    """
    Returns MFCC matrix from mp3 url
    """
    y, sr = load_mp3_from_url(mp3_url)
    M = librosa.feature.mfcc(y = y, sr = sr)
    return M


