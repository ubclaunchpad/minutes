from minutes import Minutes
import test.config as c


def test_train():
    minutes = Minutes(ms_per_observation=100)
    minutes.add_speaker(c.SPEAKER1)
    minutes.add_speaker(c.SPEAKER2)

    assert minutes.fitted is False

    # TODO: Refit and save base model, then run this.
    # minutes.fit()
