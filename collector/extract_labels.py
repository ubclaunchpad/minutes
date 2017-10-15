import logging
import re
import time

from lxml import etree
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_labels(xml, left_delim, right_delim,
                   samples_per_observation,
                   sample_rate,
                   interior='[A-Z]',
                   discard='$^'):
    """Given an XML transcription of a youtube conversation,
    returns an array of binary variables specifying whether
    a speaker is speaking.

    Args:
        xml_file (str): xml string to be parsed
        sample_rate (int): sample rate of audio that goes with xml_file
        deliminator: deliminator for speaker labels.
                    e.g. if we have "[M] Something something talking blah blah"
                    then deliminator is a regex "\[[A-Z]\]". (match open square
                    bracket, any single letter, close square bracket.)
        discard: discard any lines that contain this regex (e.g., if they label
                    sound effects in a consistant way, we can get rid of
                    those pretty easily using this param.)
    """
    # Profile time spent in function.
    start_profile = time.time()

    # Open xml and get xml tree.
    tree = etree.fromstring(xml)
    starts = tree.xpath("text/@start")
    texts = [x.text.strip("\n") for x in tree.xpath("text")]
    durs = tree.xpath("text/@dur")

    # Basic constants.
    num_seconds = float(starts[-1]) + float(durs[-1])
    num_samples = int(num_seconds * sample_rate)
    num_obs = num_samples // samples_per_observation
    logger.info('Number of seconds: {}'.format(num_seconds))
    logger.info('Number of samples: {}'.format(num_samples))
    logger.info('Number of observations: {}'.format(num_obs))

    # Convert starts and durs to samples (our unit of basic time).
    starts = [int(float(i) * sample_rate) for i in starts]
    durs = [int(float(i) * sample_rate) for i in durs]

    # Produce regexes.
    re_discard = re.compile(discard)

    # Prepend with slash if not empty string.
    left_fmt = '\\' + left_delim if left_delim else left_delim
    right_fmt = '\\' + right_delim if right_delim else right_delim
    pattern = r"{}({}){}".format(left_fmt, interior, right_fmt)
    re_delim = re.compile(pattern)

    # List of all possible speakers.
    speakers = set()
    for text in texts:
        for speaker in re_delim.findall(text):
            speakers.add(speaker)
    logger.info('Speakers found: {}'.format(speakers))

    # Map speakers to integers.
    speaker_dict = {
        speaker: i
        for i, speaker in enumerate(
            iter(speakers)
        )
    }

    # Produce result matrix.
    labels = np.zeros((num_obs, len(speakers)), dtype=np.int32)

    # Reference to the current speaker
    current = None

    # For each text, update the corresponding appropriate rows in `labels'
    # with 1's if the speaker is speaking during them. Otherwise, put
    # zeros, or -1 if the row is to be discarded.
    for start_sample, dur, text in zip(starts, durs, texts):

        # Find our window.
        start_obs = start_sample // samples_per_observation
        end_obs = (start_sample+dur) // samples_per_observation

        # Current speaker may exist in this row or previous.
        current = update_current_speaker(current, text, re_delim)

        # If discard, leave -1's so we can remove rows later.
        if not valid_text(text, re_delim, re_discard):
            labels[start_obs:end_obs, :] = -1
        elif current:
            labels[start_obs:end_obs, speaker_dict[current]] = 1

    logger.info('Result shape: {}'.format(labels.shape))
    logger.info('Expected shape: ({},{})'.format(num_obs, len(speakers)))
    logger.info('Time spent: {}'.format(time.time() - start_profile))

    return labels


def valid_text(text, re_delim, re_discard):
    """There are 3 cases where text is not valid:
        1. Multiple speakers observed in text.
        2. Text doesn't start with a speaker, but has speaker in it.
        3. The discard regex appears.
    """
    speakers = re_delim.findall(text)
    on_start = re_delim.match(text)
    discard = re_discard.search(text)

    if len(speakers) > 1:
        return False
    if not on_start and speakers:
        return False
    if discard:
        return False
    return True


def update_current_speaker(current, text, re_delim):
    """Returns current speaker based on text, previous current speaker,
    deliminator, and discard
    """
    try:
        return re_delim.findall(text)[-1]
    except IndexError:
        return current