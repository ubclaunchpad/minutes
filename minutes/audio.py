import sys

import librosa
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wav
from sklearn import preprocessing

SPECTROGRAM_HEIGHT = 1024
NUM_CHANNELS = 3


class Audio:

    def __init__(self, audio_loc):
        self.data, self.rate = wav.read(audio_loc)

    def samples_per_observation(self, ms_per_observation):
        """Converts ms per observation into samples per observation, given
        the internal sample rate of the audio.

        Arguments:
            ms_per_observation {int} -- The number of desired ms per obs.

        Returns:
            {int} -- The samples per observation.
        """
        return int(self.rate * ms_per_observation / 1000.)

    def get_spectrograms(self, ms_per_observation, verbose=False):
        """Converts a internal table of raw audio audio phrases into with
        one spectrogram per row.

        Arguments:
            ms_per_observation {int} -- The number of desired ms per obs.

        Returns:
            np.array -- An array of shape N x 1024 x 32 x 3; spectrograms.
        """
        # Reshape data into observations.
        d = self.samples_per_observation(ms_per_observation)
        N = len(self.data) // d
        data = self.data.reshape((N, d))

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

        # Choose a colormap.
        convert = plt.get_cmap(cm.jet)
        # TODO: Dynamically compute spectrogram width, 32 = 1000ms.
        imgs = np.zeros((data.shape[0], SPECTROGRAM_HEIGHT, 32, NUM_CHANNELS))

        for i in range(0, data.shape[0]):
            X = librosa.stft(data[i].astype(float))
            Xdb = librosa.amplitude_to_db(X)
            Xdb = min_max_scaler.fit_transform(Xdb)
            numpy_output_static = convert(Xdb)[:, :, :3]

            # Spectrograms are upside down, flip them.
            imgs[i] = np.flip(numpy_output_static, 0)
            if verbose:
                sys.stdout.write("Progress: %d%%   \r" % (100 * i / N))
                sys.stdout.flush()
        return imgs
