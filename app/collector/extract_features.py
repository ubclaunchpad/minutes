from python_speech_features import mfcc, fbank, logfbank, ssc
import numpy as np

def extract_features(observations, sample_rate):
    """observations: array of observations (see output of `extract_observations.py`).
       sample_rate: sample rate
       Each row of output is converted to features from `observations` using `python_speech_features`.
    """   
    # TODO: add comment to indicate which part of output are which features
    return np.apply_along_axis(\
         lambda x : \
            np.concatenate(\
                  (np.ndarray.flatten(mfcc(x, sample_rate)), \
                   np.ndarray.flatten(fbank(x, sample_rate)), \
                   np.ndarray.flatten(logfbank(x, sample_rate)), \
                   np.ndarray.flatten(ssc(x, sample_rate))),
                  axis=1), \
         1, \
         observations)
