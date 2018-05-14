import pytest

from minutes.base import BaseModel
import test.config as c


def test_add_speakers():
    model = BaseModel('taco')
    assert len(model.speakers) == 0

    model.add_speaker(c.SPEAKER1)
    assert len(model.speakers) == 1

    model.add_speaker(c.SPEAKER2)
    assert len(model.speakers) == 2

    model = BaseModel('taco')
    assert len(model.speakers) == 0
    model.add_speakers([c.SPEAKER1, c.SPEAKER2])
    assert len(model.speakers) == 2


def test_unique_speakers():
    model = BaseModel('taco')
    model.add_speaker(c.SPEAKER1)

    # Should bark about duplicate speakers.
    with pytest.raises(LookupError):
        model.add_speaker(c.SPEAKER1)


def test_generate_training_data():
    # Using small audio files to keep tests fast; set ms_per_observation
    # low to generate an adequate number of training observations.
    model = BaseModel('taco', ms_per_observation=100)
    model.add_speaker(c.SPEAKER1)
    model.add_speaker(c.SPEAKER2)

    Xtr, Xt, ytr, yt = model._generate_training_data()

    # Built spectrograms with shape (129, 19), one-hot labels.
    assert Xtr.shape == (218, 129, 7)
    assert ytr.shape == (218, 2)
    assert Xt.shape == (108, 129, 7)
    assert yt.shape == (108, 2)


def test_train():
    model = BaseModel('taco', ms_per_observation=100)
    model.add_speaker(c.SPEAKER1)
    model.add_speaker(c.SPEAKER2)
    assert not model.fitted
    model.fit()
    assert model.fitted
