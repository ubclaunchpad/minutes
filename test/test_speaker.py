import os

from minutes import Speaker
import test.config as config

SPEAKER1_AUDIO = os.path.join(config.FIXTURE_DIR, 'sample1.wav')
SPEAKER2_AUDIO = os.path.join(config.FIXTURE_DIR, 'sample2.wav')


def test_speaker_add_samples():
    speaker1 = Speaker('speaker1')
    speaker1.add_audio(SPEAKER1_AUDIO)
    assert len(speaker1.audio) == 1

    speaker1.add_audio(SPEAKER2_AUDIO)
    assert len(speaker1.audio) == 2


def test_speaker_observations():
    pass
