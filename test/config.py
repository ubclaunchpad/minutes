import contextlib
import os
import shutil
import tempfile


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


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def tempdir():
    """Creates a temporary directory for files used during tests."""
    dirpath = tempfile.mkdtemp()
    with cd(dirpath, lambda: shutil.rmtree(dirpath)):
        yield dirpath
