# minutes

[![ZenHub](https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.com)

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)

[![Coverage Status](https://coveralls.io/repos/github/ubclaunchpad/minutes/badge.svg)](https://coveralls.io/github/ubclaunchpad/minutes)

Audio speaker diarization library.

## Under Construction!

## :running: Development

Dependencies are managed using a `Pipfile` and [Pipenv](https://github.com/pypa/pipenv):

```bash
pipenv install
pipenv shell
```

## Testing

```bash
pytest --cov=minutes -vvv test
```

## Example Usage

```python
from minutes import Speaker, Minutes

minutes = Minutes(ms_per_observation=500, model='cnn')

# Create some speakers with some audio.
speaker1 = Speaker('speaker1')
speaker1.add_audio('path/to/audio1.wav')

speaker2 = Speaker('speaker2')
speaker2.add_audio('path/to/audio2.wav')

# Add speakers to the model.
minutes.add_speakers([speaker1, speaker2])

# Fit the model.
minutes.fit()
result = minutes.predict()
```
