from minutes import Minutes
import test.config as c


def test_train():
    for model_name in Minutes.models:
        minutes = Minutes(model=model_name, ms_per_observation=3000)
        minutes.add_speaker(c.SPEAKER1)
        minutes.add_speaker(c.SPEAKER2)

        assert minutes.fitted is False
        minutes.fit()
        assert minutes.fitted is True
