import test.config as c


def test_speaker_add_samples():
    assert len(c.SPEAKER1.audio) == 1


def test_speaker_equality():
    assert c.SPEAKER1 == c.SPEAKER1
    assert c.SPEAKER2 != c.SPEAKER1
