import numpy as np

from minutes.audio import Audio


class Speaker:
    """A speaker in a Minutes conversation."""

    def __init__(self, name):
        """Create a new speaker for inclusion in a conversation.

        Arguments:
            name {str} -- The speakers name.
        """
        self.name = name
        self.audio = []

    def add_audio(self, audio_loc):
        """Add an audio sample to this speaker.

        Arguments:
            audio_loc {str} -- Location of the audio sample.
        """
        self.audio += Audio(audio_loc),

    def get_observations(self, **preprocessing_params):
        raw, processed = [], []
        for a in self.audio:
            r, p = a.get_observations(**preprocessing_params)
            raw += r,
            processed += p,
        return np.concatenate(raw), np.concatenate(processed)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))
