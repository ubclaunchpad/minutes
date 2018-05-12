import numpy as np
import scipy.signal as signal
from scipy.io import wavfile as wav


class Audio:

    def __init__(self, audio_loc):
        self.rate, self.data = wav.read(audio_loc)

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
            np.array -- An array of spectrograms, one per row. The width
            of each spectrogram depends on the ms_per_observation,
            The number of rows depends on the length of the audio file
            and the ms per observations.

        TODO: Provide different spectrogram creation options; for now
        mode='phase' is all you get.
        """
        d = self.samples_per_observation(ms_per_observation)
        N = len(self.data) // d

        # Merge stereo.
        if self.data.shape[0] > 1:
            data = np.mean(self.data, axis=1)
        else:
            data = self.data.copy()

        # Truncate last (partial) observation.
        data = data[:N * d]

        if verbose:
            t = len(self.data) - (N * d)
            print('Truncating {} bytes from end of sample'.format(t))

        # Reshape for processing into spectrograms.
        data = data.reshape((N, d))

        def spec_from_row(row):
            _, _, Sxx = signal.spectrogram(data, mode='phase')
            return Sxx

        # This is very slow! Perhaps some logging?
        rows = (spec_from_row(row for row in data))
        return np.array([x for x in rows])
