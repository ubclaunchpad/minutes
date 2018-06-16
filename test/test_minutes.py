import numpy as np

from minutes import Minutes, Conversation
import test.config as c


def test_train():
    for model_name in Minutes.parents:
        minutes = Minutes(parent=model_name)
        minutes.add_speaker(c.SPEAKER1)
        minutes.add_speaker(c.SPEAKER2)

        assert minutes.fitted is False
        minutes.fit()
        assert minutes.fitted is True


def test_parents():
    assert Minutes.parents == ['cnn']


def test_phrases():
    for model_name in Minutes.parents:
        minutes = Minutes(parent=model_name)
        minutes.add_speaker(c.SPEAKER1)
        minutes.add_speaker(c.SPEAKER2)
        minutes.fit()

        # Predict new phrases (make sure we ony predict once per obs)
        conversation = Conversation(c.CONVERSATION_AUDIO, minutes)
        raw, _ = conversation.get_observations(**minutes.preprocessing_params)
        assert len(conversation.phrases) == len(raw)

        # Make sure we predicted some subset of the acceptable values.
        names = [p.speaker.name for p in conversation.phrases]
        expected = {'speaker1', 'speaker2'}
        assert set(np.unique(names)) <= expected
