# minutes

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)
[![Coverage Status](https://coveralls.io/repos/github/ubclaunchpad/minutes/badge.svg)](https://coveralls.io/github/ubclaunchpad/minutes)

> Jotting things down, so you don't have to.

![Spectrogram](/.static/spec.png)

Minutes is a speaker diarisation library. [Speaker diarisation](https://en.wikipedia.org/wiki/Speaker_diarisation) is the process
of identifying different speakers in an audio segment. It is useful for
making transcriptions of conversations meaningful by tagging homogenous
sections of the conversation with the appropriate speaker.

For more information about Minutes, and how it works, check out our [Medium
post](https://medium.com/ubc-launch-pad-software-engineering-blog/speaker-diarisation-using-transfer-learning-47ca1a1226f4)!


## :point_up: Installation

Requires Python 3.6!

```bash
# Currently we recommend running using the pipenv shell below.
python setup.py install
```

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

# Collect a new conversation for prediction.
conversation = Conversation('/path/to/conversation.wav')

# Create phrases from the conversation.
phrases = minutes.phrases(conversation)
```
