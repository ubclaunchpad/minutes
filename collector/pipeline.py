import json
import logging
import os
import re

import numpy as np
import pandas as pd
import python_speech_features as psf
from scipy.io import wavfile
import xmltodict


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def flatten(l):
    """Flattens a list of lists to a single list.

    Args:
        l a nested list.
    """
    return [item for sublist in l for item in sublist]


def get_speakers(t, left, right, interior='[A-Za-z]+'):
    """Returns the speakers in the string t as specified by
    the delimiters `left' and `right'.

    Args:
        t (str): a phrase from the conversation (perhaps
            with speakers in it).
        left (str): a left delimiter such as '[' or ''.
        right (str): a right delimiter such as ']' or ':'.
        interior (str) a regex to capture the text between
            the left and right delimiters.
    """
    # Prepend with slash if not empty string.
    left_fmt = '\\' + left if left else left
    right_fmt = '\\' + right if right else right

    # There is something fancier we can do here, but this seems
    # to be generalizing well.
    pattern = r"{}({}){}".format(left_fmt, interior, right_fmt)
    return re.findall(
        pattern=pattern,
        string=t
    )


def extract_labels(xml, left_delim, right_delim,
                   interior='[A-Za-z]+', rate=44100):
    """Given an XML transcription of a youtube conversation,
    returns a dataframe of binary variables specifying whether
    a speaker is speaking.

    Args:
        xml (str): xml input encoded as utf-8.
        left_delim (str): the left delimeter on a speaker tag.
        right_delim (str): the right delimiter on a speaker tag.
        rate (int) represents the sample rate of the audio in Hz;
            if you want to split the result by seconds let rate=1,
            if you want to use the ideal sampling rate given
            by the Nyquist Theorem, use 44100 (rather slow).
    """

    assert 0 < rate <= 44100, "rate must be on [0, 441000]"

    # Format the xml into a dataframe.
    d = xmltodict.parse(xml_input=xml)
    df = pd.DataFrame(d['transcript']['text'])
    df['@start'] = pd.to_numeric(df['@start'])
    df['@dur'] = pd.to_numeric(df['@dur'])

    # Build up a set of speakers.
    speakers = set(
        flatten([
            get_speakers(
                t=t,
                left=left_delim,
                right=right_delim,
                interior=interior
            ) for t in df['#text'].tolist()
        ])
    )

    logger.info('Found speakers: {}'.format(','.join(speakers)))

    assert speakers, "No speakers were found in transcription"

    # Find out how many samples we'll need to generate.
    length_sec = df.tail(1)['@start'] + df.tail(1)['@dur']
    num_samples = int(length_sec * rate)

    # Dataframe to hold results.
    result = pd.DataFrame(
        np.nan,
        index=range(num_samples),
        columns=speakers
    )

    # Each record will be used to fill some samples in result.
    for i, record in enumerate(df.to_dict(orient='records')):

        pct = 100.0 * i / len(df)
        if int(pct) % 25 == 0:
            logger.info('{}% of texts analyzed'.format(pct))

        # Find the window start and end (in samples).
        start = int(record['@start'] * rate)
        end = start + int(record['@dur'] * rate)

        # Find out which speakers are speaking
        # in this sample.
        speakers_in_record = get_speakers(
            t=record['#text'],
            left=left_delim,
            right=right_delim,
            interior=interior
        )

        # See if each speaker is speaking during window.
        if speakers_in_record:
            for s in speakers_in_record:
                # Leave no's as nans, backfill later.
                # TODO: This does not split the phrase by speaker,
                # it just pretends "all speakers are speaking right
                # now." Not the best, do fix.
                speaking = 1 if s in speakers_in_record else np.nan
                result.loc[start: end, s] = speaking
                previous_speaker = s
        else:
            # Speaker is previous speaker.
            # This breaks if there is no speaker in
            # the first phrase. For now, let's ignore the
            # first phrase (its normally something like
            # [music plays]).
            try:
                result.loc[start:end, previous_speaker] = 1
            except UnboundLocalError:
                pass

    # Fill the rest with 0's.
    return result.fillna(0)


def extract_features(audio_file, rate):
    """Given `audio' as a numpy array, produces audio feature
    vectors at a rate of `rate' Hz.

    Note:
        If there are n seconds of audio date, there will be
        rate * n feature vectors.

    Args:
        audio_file (str): Filename and location of audio file.
        rate (int): The feature extraction rate.

    Returns:
        df (pd.DataFrame): A dataframe of feature vectors.
    """
    _, data = wavfile.read(audio_file)

    # Pull features.
    logger.info('Extracting mfcc...')
    mfcc = pd.DataFrame(psf.mfcc(data, rate))
    logger.info('Extracting logfbank...')
    logfbank = pd.DataFrame(psf.logfbank(data, rate))
    logger.info('Extracting fbank...')
    fbank = pd.DataFrame(psf.fbank(data, rate))
    logger.info('Extracting ssc...')
    ssc = pd.DataFrame(psf.ssc(data, rate))

    # Name columns.
    mfcc.columns = [i + '_mfcc' for i in mfcc.columns]
    logfbank.columns = [i + '_logfbank' for i in logfbank.columns]
    fbank.columns = [i + '_fbank' for i in fbank.columns]
    ssc.columns = [i + '_ssc' for i in ssc.columns]

    # Concatenate results (column-bind).
    df = pd.concat([mfcc, logfbank, fbank, ssc], axis=1)

    return df


def build(sample_id):
    """Produces a training datset from YouTube audio and
    XML transcripts.

    Args:
        sample_id (str): The youtube ID of this sample.
            In the folder with the name `sample_id', the
            following items must appear:
                1) An XML file named sample_id.xml
                2) A wave file named sample_id.wav
                3) A json file named sample_id.json
    """

    logger.info('Working on id {}'.format(sample_id))

    # Get parameters from results folder.
    json_location = os.path.join(sample_id, sample_id + '.json')
    audio_file = os.path.join(sample_id, sample_id + '.wav')
    xml_file = os.path.join(sample_id, sample_id + '.xml')

    with open(json_location, 'r') as infile:
        info = json.load(infile)

    logger.info('Using parameters {}'.format(info))

    left_delim = info['left_delim']
    right_delim = info['right_delim']
    interior = info['interior']
    rate = info['rate']

    # Grab features.
    logger.info('Extracting features...')
    features = extract_features(audio_file, rate)

    # Bring in xml file and pull labels.
    with open(xml_file, 'r') as xml_infile:
        logger.info('Extracting labels...')
        labels = extract_labels(
            xml=xml_infile.read(),
            left_delim=left_delim,
            right_delim=right_delim,
            interior=interior,
            rate=44100
        )

    # Column-bind results.
    data = pd.concat([features, labels], axis=1)

    # Dump to csv.
    output = os.path.join(sample_id, sample_id + ".csv")

    with open(output, 'w') as outfile:
        data.to_csv(outfile)


# Testing code - remove before merging.
if __name__ == '__main__':
    build('Xz3btMhdQ6Y')
