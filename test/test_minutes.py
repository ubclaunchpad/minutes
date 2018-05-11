import os

import pytest

from minutes import Speaker, Minutes
import test.config as config

SPEAKER1_AUDIO = os.path.join(config.FIXTURE_DIR, 'sample1.wav')
SPEAKER2_AUDIO = os.path.join(config.FIXTURE_DIR, 'sample2.wav')


def test_add_speakers():
    speaker1 = Speaker('speaker1')
    minutes = Minutes()
    minutes.add_speaker(speaker1)
    assert len(minutes.speakers) == 1

    speaker2 = Speaker('speaker2')
    minutes.add_speaker(speaker2)
    assert len(minutes.speakers) == 2


def test_unique_speakers():
    speaker1 = Speaker('speaker1')
    minutes = Minutes()
    minutes.add_speaker(speaker1)

    # Should bark about speakers.
    with pytest.raises(LookupError):
        minutes.add_speaker(speaker1)


def test_generate_training_data():
    minutes = Minutes()
    speaker1 = Speaker('speaker1')
    speaker2 = Speaker('speaker2')
    minutes.add_speaker(speaker1)
    minutes.add_speaker(speaker2)

    # TODO: Create training data and assert shape makes sense.
    # data = minutes._generate_training_data()
