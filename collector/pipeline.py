import json
import logging
import os

import numpy as np
from scipy.io import wavfile as wav

from extract_labels import extract_labels
from extract_observations import extract_observations


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Hyper paramters.
SAMPLES_PER_OBSERVATION = 500


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

    # Bring in audio.
    sample_rate, signal = wav.read(audio_file)

    with open(json_location, 'r') as infile:
        info = json.load(infile)

    logger.info('Using parameters {}'.format(info))

    left_delim = info['left_delim']
    right_delim = info['right_delim']
    interior = info['interior']

    # Bring in xml file and pull labels.
    with open(xml_file, 'r') as xml_infile:
        logger.info('Extracting labels...')
        labels = extract_labels(
            xml=xml_infile.read().encode('ascii'),
            left_delim=left_delim,
            right_delim=right_delim,
            interior=interior,
            sample_rate=sample_rate,
            samples_per_observation=SAMPLES_PER_OBSERVATION
        )

    # Grab features (this crashes due to memory at the moment).
    logger.info('Extracting features...')
    features = extract_observations(None, None, None)

    # Column-bind results.
    logger.info('Feature shape {}'.format(features.shape))
    logger.info('Label shape {}'.format(labels.shape))

    # Dump CSV's.
    for fname, tbl in [
        ('features', features),
        ('labels', labels),
    ]:
        output = os.path.join(
            sample_id, sample_id + "-{}.csv".format(fname))
        logger.info(
            'Writing {} to CSV (this can take a minute)...'.format(
                fname))
        np.savetxt(output, tbl, delimiter=',')
    
    logger.info('Completed succesfully.')


# Testing code - remove before merging.
if __name__ == '__main__':
    build('Xz3btMhdQ6Y')
