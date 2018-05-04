from audio import Audio


class Conversation:

    def __init__(self, audio_loc, speakers):
        """Create a new conversation from audio sample.

        Arguments:
            audio_loc {str} -- The absolute location of an audio conversation
            sample.
            ms_per_observation {int} -- Number of milliseconds per observation.
            speakers {List[Speaker]} -- A list of speakers included in this
            conversation.
        """
        self.speakers = speakers
        self.audio = Audio(audio_loc)

    def get_observations(self, ms_per_observation=1000, verbose=False):
        """Converts the conversation audio sample into an n x d matrix of
        observations.

        Keyword Arguments:
            verbose {bool} -- (default: {False})
            ms_per_observation {int} -- (default: {False})

        Returns:
            np.array -- An n x d matrix of observations.
        """
        return self.audio.get_spectrograms(ms_per_observation, verbose)

    def phrases(self, model):
        """Given a trained model, recreates phrases from the internal
        audio sample.

        Arguments:
            model {keras.model} -- A model trained on the speakers in
            the converstation.
        """
        pass  # TODO
