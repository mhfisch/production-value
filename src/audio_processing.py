import urllib.request
from os import path
from pydub import AudioSegment
import librosa
from scipy import signal


"""
-------
LOADING
-------
"""


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


"""
----------------
BANDPASS FILTERS
----------------
"""

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a 

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='bandpass', analog=False)
    return b, a 

def butter_bandpass_filter(data, cutoff, fs, order=5):
    """
    cutoff must be a 2 element numpy array
    """
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandstop(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='bandstop', analog=False)
    return b, a 

def butter_bandstop_filter(data, cutoff, fs, order=5):
    """
    cutoff must be a 2 element numpy array
    """
    b, a = butter_bandstop(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



"""
---------------
MFCC GENERATORS
---------------
"""

def mfcc_highpass(y, sr):
    y_highpass = butter_highpass_filter(y,800,sr)
    M = librosa.feature.mfcc(y_highpass)
    return M

def mfcc_harmonic(y, sr):
    D = librosa.stft(y)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=2)
    y_harmonic = librosa.core.istft(D_harmonic)
    M = librosa.feature.mfcc(y_harmonic)
    return M

def mfcc_percussive(y, sr):
    D = librosa.stft(y)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=2)
    y_percussive = librosa.core.istft(D_percussive)
    M = mfcc = librosa.feature.mfcc(y_percussive)
    return M
    
def mfcc_shuffle(y, sr):
    # Calculate beats
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_samples = librosa.frames_to_samples(beat_frames)
    intervals = librosa.util.frame(beat_samples, frame_length=2, hop_length=1).T

    # Shuffle intervals
    interval_idx = list(range(len(intervals)))
    np.random.shuffle(interval_idx)
    shuffled_intervals = intervals[interval_idx]
    y_shuffle = librosa.effects.remix(y, shuffled_intervals)  
    M = librosa.feature.mfcc(y_shuffle)
    return M



"""
---------------------
MISC AUDIO PROCESSING
---------------------
"""
def shuffle_beats(y, sr):
    # Calculate beats
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_samples = librosa.frames_to_samples(beat_frames)
    intervals = librosa.util.frame(beat_samples, frame_length=2, hop_length=1).T

    # Shuffle intervals
    interval_idx = list(range(len(intervals)))
    np.random.shuffle(interval_idx)
    shuffled_intervals = intervals[interval_idx]

    y_shuffle = librosa.effects.remix(y, shuffled_intervals)
    return y_shuffle