import glob
import os

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import soundfile as sf


class Audio:

    def __init__(self, audio_loc):
        if os.path.isdir(audio_loc):
            audio, rates = [], []
            wav_files = glob.glob(audio_loc + '/**/*.wav', recursive=True)

            # Collect all data and concatenate it.
            for file in wav_files:
                data, rate = sf.read(file)
                audio += data,
                rates += rate,

            self.data = np.concatenate(audio)

            # Set rate as mode of rates for now.
            # TODO: Resample audio?
            self.rate = stats.mode(rates)[0][0]
        else:
            self.data, self.rate = sf.read(audio_loc)

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
        """
        d = self.samples_per_observation(ms_per_observation)
        N = len(self.data) // d

        # Merge stereo if necessary.
        if len(self.data.shape) == 1:
            data = self.data.copy()
        else:
            data = np.mean(self.data, axis=1)

        # Truncate last (partial) observation.
        data = data[:N * d]

        if verbose:
            t = len(self.data) - (N * d)
            print('Truncating {} bytes from end of sample'.format(t))

        # Reshape for processing into spectrograms.
        data = data.reshape((N, d))

        def spec_from_row(row):
            _, _, Sxx = signal.spectrogram(row)
            return Sxx

        # This is very slow! Perhaps some logging?
        rows = (spec_from_row(row) for row in data)
        return np.array([x for x in rows])
