# minutes

[![ZenHub](https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.com)

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)

[![Coverage Status](https://coveralls.io/repos/github/ubclaunchpad/minutes/badge.svg)](https://coveralls.io/github/ubclaunchpad/minutes)

Audio speaker diarization library. 

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
from minutes import Speaker, Minutes, Conversation

minutes = Minutes(parent='cnn')

# Create some speakers, add some audio.
s1, s2 = Speaker('s1'), Speaker('s2')
s1.add_audio('path/to/audio1')
s2.add_audio('path/to/audio2')

# Add speakers to the model.
minutes.add_speakers([s1, s2])

# Fit the model.
minutes.fit()

# Predict against a new conversation had by speakers s1 and s2.
conversation = Conversation('/path/to/conversation.wav', minutes)
```
