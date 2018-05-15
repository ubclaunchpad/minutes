import os

from minutes import Speaker

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(TEST_DIR, '..')
FIXTURE_DIR = os.path.join(TEST_DIR, 'fixtures')

SPEAKER1_AUDIO = os.path.join(FIXTURE_DIR, 'sample1.wav')
SPEAKER2_AUDIO = os.path.join(FIXTURE_DIR, 'sample2.wav')

# Load speaker audio just once for all tests.
SPEAKER1 = Speaker('speaker1')
SPEAKER1.add_audio(SPEAKER1_AUDIO)

SPEAKER2 = Speaker('speaker2')
SPEAKER2.add_audio(SPEAKER2_AUDIO)
