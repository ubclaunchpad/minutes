from python_speech_features import mfcc
from python_speech_features import logfbank
import numpy as np

def extract_features(observations, sample_rate):
    """observations: array of observations (see output of `extract_observations.py`).
       sample_rate: sample rate
       Each row of output is converted to features from `observations` using `python_speech_features`.
    """   
    # make features array 
    # TODO: fix placeholder for width
    # width = 26 # number of columns of output array, calculated based on what features python_speech_features gives
    # features = np.zeros(observations.shape[0], width) 
    # convert rows to features 

    return np.apply_along_axis(lambda x : mfcc(x, sample_rate)[0], 1, observations)
    # for i in range(observations.shape[0]):
    #    features[i, :] = mfcc(observations[i, :], sample_rate)
    # return features array
    # return features
