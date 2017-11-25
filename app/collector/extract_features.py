import sys
import logging

import python_speech_features as psf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Size of the FFT window.
NFFT = 2048


def extract_features(observations, sample_rate, new_logger=None):
    """Converts each row of the raw observations to features.
    
    Args:
        observations (ndarray): array of observations (see output of
            `extract_observations.py`).
       sample_rate (int): sample rate as reported by scipy.io.wavfile
       logger (logger): pass in a logger from another application.

    Returns:
        arr (np.ndarray): An array with the same number of rows of
            as that in `observations`. Number of columns is a function
            of the number of output features produced by the mfcc,
            ssc and logfbank feature extractors.
    """

    if new_logger is not None:
        logger = new_logger

    num_rows = observations.shape[0]

    def extract_row(row, row_num):
        sys.stdout.write("Row out of %d: %d   \r" % (num_rows, row_num))
        sys.stdout.flush()

        # Average the two signals (they are interleved).
        mono_signal = np.mean(row.reshape(-1, 2), axis=1)

        # Produce features.
        mfcc_features = np.hstack(
            psf.mfcc(mono_signal, sample_rate, nfft=NFFT))
        logfbank_features = np.hstack(
            psf.logfbank(mono_signal, sample_rate, nfft=NFFT))
        ssc_features = np.hstack(
            psf.ssc(mono_signal, sample_rate, nfft=NFFT))

        # Concatenate results.
        return np.concatenate(
            (mfcc_features, logfbank_features, ssc_features))

    # Would use np.apply_along_axis, but no speedup and we need
    # the row index for logging.
    return np.array([
        extract_row(row, i)
        for i, row in enumerate(observations)
    ])
