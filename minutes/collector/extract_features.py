import logging

import python_speech_features as psf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Size of the FFT window.
NFFT = 2048


def extract_features(observations, sample_rate):
    """Converts each row of the raw observations to features.
    
    Args:
        observations (ndarray): array of observations (see output of
            `extract_observations.py`).
       sample_rate (int): sample rate as reported by scipy.io.wavfile

    Returns:
        arr (np.ndarray): An array with the same number of rows of
            as that in `observations`. Number of columns is a function
            of the number of output features produced by the mfcc,
            ssc and logfbank feature extractors.
    """

    num_rows = observations.shape[0]

    def extract_row(row, row_num):
        # Yes. this is silly, but it works very well.
        prop_done = 100 * row_num / num_rows
        if prop_done % 2 == 0 and prop_done > 1:
            logger.info(f'{prop_done}% done.')

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
