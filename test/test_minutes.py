import numpy as np

from minutes import Minutes
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

        # Predict new phrases.
        phrases = minutes.phrases(c.CONVERSATION)
        speakers = [p.speaker.name for p in phrases]

        ms_per_obs = minutes.parent.ms_per_observation
        num_obs = c.CONVERSATION.get_observations(ms_per_obs)
        assert len(phrases) == len(num_obs)
        assert sorted(list(np.unique(speakers))) == ['speaker1', 'speaker2']
