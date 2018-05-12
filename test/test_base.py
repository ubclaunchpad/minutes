import pytest

from minutes import Minutes
import test.config as c


def test_add_speakers():
    minutes = Minutes()
    assert len(minutes.speakers) == 0

    minutes.add_speaker(c.SPEAKER1)
    assert len(minutes.speakers) == 1

    minutes.add_speaker(c.SPEAKER2)
    assert len(minutes.speakers) == 2

    minutes = Minutes()
    assert len(minutes.speakers) == 0
    minutes.add_speakers([c.SPEAKER1, c.SPEAKER2])
    assert len(minutes.speakers) == 2


def test_unique_speakers():
    minutes = Minutes()
    minutes.add_speaker(c.SPEAKER1)

    # Should bark about duplicate speakers.
    with pytest.raises(LookupError):
        minutes.add_speaker(c.SPEAKER1)


def test_generate_training_data():
    # Using small audio files to keep tests fast; set ms_per_observation
    # low to generate an adequate number of training observations.
    minutes = Minutes(ms_per_observation=100)
    minutes.add_speaker(c.SPEAKER1)
    minutes.add_speaker(c.SPEAKER2)

    Xtr, Xt, ytr, yt = minutes._generate_training_data()

    # Built spectrograms with shape (129, 19), one-hot labels.
    assert Xtr.shape == (131, 129, 19)
    assert ytr.shape == (131, 2)
    assert Xt.shape == (66, 129, 19)
    assert yt.shape == (66, 2)
