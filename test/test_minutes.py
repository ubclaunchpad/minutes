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
