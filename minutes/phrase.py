

class Phrase:
    def __init__(self, observation, speaker):
        """[summary]

        Arguments:
            observation {np.array} -- 1 dimensional audio sample.
            speaker {[type]} -- Predicted Speaker.
        """
        self.observation = observation
        self.speaker = speaker
