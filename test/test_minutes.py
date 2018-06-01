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
        print(conversation.phrases)

        # Make sure we ony predicted on speaker 1 and 2.
        names = [p.speaker.name for p in conversation.phrases]
        assert sorted(list(np.unique(names))) == ['speaker1', 'speaker2']
