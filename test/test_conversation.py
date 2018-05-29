import test.config as c


def test_conversation_observation_shape():
    obs = c.CONVERSATION.get_spectrograms(50)
    assert obs.shape == (654, 129, 3)
