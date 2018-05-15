import numpy as np

from minutes.audio import Audio
import test.config as c


def test_add_audio():
    audio = Audio(c.SPEAKER1_AUDIO)
    assert audio.rate == 16000


def test_samples_per_observation():
    audio = Audio(c.SPEAKER1_AUDIO)
    assert audio.samples_per_observation(1000) == audio.rate


def test_get_spectrograms():
    audio = Audio(c.SPEAKER1_AUDIO)
    spec = audio.get_spectrograms(3000)
    assert spec.shape == (5, 129, 214)
    assert spec.dtype == np.float64
