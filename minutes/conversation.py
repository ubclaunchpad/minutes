from minutes.audio import Audio


class Phrase:
    def __init__(self, observation, speaker):
        """A phrase in a conversation, characterized by an audio segment
        and a speaker.

        Arguments:
            observation {np.array} -- 1 dimensional audio sample.
            speaker {Speaker} -- The inferred speaker for the audio segment.
        """
        self.observation = observation
        self.speaker = speaker


class Conversation(Audio):

    def __init__(self, audio_loc, model):
        """Create a new conversation from audio sample.

        Arguments:
            audio_loc {str} -- The absolute location of an audio conversation
            sample.
            model {Minutes} -- A model trained on speakers within this
            conversation.
        """
        self.model = model
        super().__init__(audio_loc)

        # Predict against the conversation spectrograms.
        raw, X_hat = self.get_observations(**model.preprocessing_params)
        y_hat = model.predict(X_hat)

        # Convert to a list of phrases.
        self.phrases = [Phrase(o, speaker) for o, speaker in zip(raw, y_hat)]
