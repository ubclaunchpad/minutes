from minutes.audio import Audio


class Conversation:

    def __init__(self, audio_loc):
        """Create a new conversation from audio sample.

        Arguments:
            audio_loc {str} -- The absolute location of an audio conversation
            sample.
        """
        self.audio = Audio(audio_loc)

    def get_spectrograms(self, ms_per_observation, verbose=False):
        """Converts the conversation audio sample into an n x d matrix of
        spectrograms.

        Keyword Arguments:
            verbose {bool} -- (default: {False})
            ms_per_observation {int} -- (default: {False})

        Returns:
            np.array -- An n x d matrix of spectrograms.
        """
        return self.audio.get_spectrograms(ms_per_observation, verbose)

    def get_observations(self, ms_per_observation, verbose=False):
        """Converts the conversation audio sample into an n x d matrix of
        observations.

        Keyword Arguments:
            verbose {bool} -- (default: {False})
            ms_per_observation {int} -- (default: {False})

        Returns:
            np.array -- An n x d matrix of observations.
        """
        return self.audio.get_observations(ms_per_observation, verbose)
