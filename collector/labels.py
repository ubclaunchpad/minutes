import json
import os
import re
import sys

import numpy as np
import requests
import pandas as pd
import xmltodict


def flatten(l):
    """
    Flattens a list of lists to a single list.

    @param l a nested list.
    """
    return [item for sublist in l for item in sublist]


def get_speakers(t, left, right, interior='[A-Za-z]+'):
    """
    Returns the speakers in the string t as specified by
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


def create_labels(xml, left_delim, right_delim,
                  interior='[A-Za-z]+', S=44100):
    """
    Given an XML transcription of a youtube conversation,
    returns a dataframe of binary variables specifying whether
    a speaker is speaking.

    Args:
        xml (str): xml input encoded as utf-8.
        left_delim (str): the left delimeter on a speaker tag.
        right_delim (str): the right delimiter on a speaker tag.
        S (int) represents the sample rate of the audio in Hz;
            if you want to split the result by seconds let S=1,
            if you want to use the ideal sampling rate given
            by the Nyquist Theorem, use 44100 (rather slow).
    """

    assert 0 < S <= 44100, "S must be on [0, 441000]"

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

    assert speakers, "No speakers were found in transcription"

    # Find out how many samples we'll need to generate.
    length_sec = df.tail(1)['@start'] + df.tail(1)['@dur']
    num_samples = int(length_sec * S)

    # Dataframe to hold results.
    result = pd.DataFrame(
        np.nan,
        index=range(num_samples),
        columns=speakers
    )

    # Each record will be used to fill some samples in result.
    for record in df.to_dict(orient='records'):

        # Find the window start and end (in samples).
        start = int(record['@start'] * S)
        end = start + int(record['@dur'] * S)

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


# Testing code - to remove.
if __name__ == '__main__':
    index = 1
    with open('ids.json', 'r') as infile:
        ids = json.loads(infile.read())
    
    youtube_info = ids[index]
    xml_path = os.path.join(
        youtube_info['id'], 
        youtube_info['id'] + '.xml'
    )

    labels = create_labels(
        xml=open(xml_path, 'r').read(),
        left_delim=youtube_info['left_delim'],
        right_delim=youtube_info['right_delim'],
        interior='[A-Z]',
        S=44100
    )

    print(labels)
